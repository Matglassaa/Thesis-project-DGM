################################################################################
# Imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import os
import argparse
from pathlib import Path
import shutil
from functools import partial
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim

# Import your Custom Dataloader
from dataloader import FaciesDataset

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks import resnet
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import R1Regularization
from voxgan.models.metrics import MSSWD, LoS

def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN Architecture 4 for Geomodelling")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory (.h5 file)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--disable_one_hot', action='store_true', help='Disable one-hot encoding (use single-channel raw indices)')
    parser.add_argument('--validation_size', type=float, default=0.1, help='Percentage of files set aside for validation (default: 0.1 for 10%)')
    return parser.parse_args()

def main():
    args = parse_args()

    ################################################################################
    # Setting
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    ################################################################################
    # Model configuration
    model_name = 'architecture_4_dcgan'
    nz = 100
    nl = (4, 6, 6)
    
    use_one_hot = not args.disable_one_hot
    nc = 3 if use_one_hot else 1
    encoding_tag = "one_hot" if use_one_hot else "no_one_hot"
    
    # Strictly matching working script: Always use Tanh for output bounds
    last_activation = nn.Tanh

    generator = partial(resnet.DeepGenerator3d,
                        nz=nz, ngf=64, nc=nc, nl=nl,
                        max_factor=16, residual_weight=1., mode='nearest', kernel_size=3,
                        layer_normalization=nn.BatchNorm3d,
                        last_layer_normalization=nn.BatchNorm3d,
                        weight_normalization=nn.utils.parametrizations.spectral_norm,
                        activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                        last_activation=last_activation,
                        use_double_conv=False, use_double_resblocks=False,
                        use_attention=False, skip_z=False, split_z=False)

    discriminator = partial(resnet.DeepDiscriminator3d,
                            ndf=64, nc=nc, nl=nl,
                            max_factor=16, residual_weight=1., kernel_size=3,
                            layer_normalization=None,
                            weight_normalization=nn.utils.parametrizations.spectral_norm,
                            activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                            use_double_conv=False, use_double_resblocks=False,
                            use_attention=False)

    optimizer_generator = partial(optim.Adam, lr=2e-4, betas=(0., 0.99))
    optimizer_discriminator = partial(optim.Adam, lr=2e-4, betas=(0., 0.99))
    
    loss_generator = nn.BCEWithLogitsLoss()
    loss_discriminator = nn.BCEWithLogitsLoss()
    
    penalty_discriminator = R1Regularization(gamma=10., num_iter=16, use_amp=True)

    ################################################################################
    # Validation
    metric_params = dict(n_levels=3,
                         n_descriptors=512,
                         descriptor_size=(3, 7, 7),
                         n_repeat=12,
                         n_proj=128,
                         padding_mode='circular', # Match working script exactly
                         combine_levels=True, 
                         batch_size=args.val_batch_size,
                         n_gpu=args.num_gpus)

    if use_one_hot:
        ms_swd_fa1 = MSSWD(**metric_params, channel=0)
        ms_swd_fa2 = MSSWD(**metric_params, channel=1)
        ms_swd_fa3 = MSSWD(**metric_params, channel=2)
        los = LoS(channel=0, batch_size=args.val_batch_size, n_gpu=args.num_gpus)
        metrics = [ms_swd_fa1, ms_swd_fa2, ms_swd_fa3, los]
    else:
        ms_swd = MSSWD(**metric_params)
        los = LoS(channel=0, batch_size=args.val_batch_size, n_gpu=args.num_gpus)
        metrics = [ms_swd, los]

    ################################################################################
    # Dataset
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    
    dataset = partial(FaciesDataset,
                      root=args.data_dir,
                      save_mapping_dir=args.output_dir,
                      use_one_hot=use_one_hot,
                      dataset_name=dataset_name,
                      num_epochs=args.epochs)

    ################################################################################
    # Training
    num_training = 1 
    
    for i in range(num_training):
        output_label = f'fluvgan_1_training_1_{model_name}_{encoding_tag}_{i + 1}'
        
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=args.output_dir,
                        output_label=output_label,
                        verbose=2,
                        num_gpus=args.num_gpus,
                        num_nodes=1,
                        distributed=False,
                        backend='nccl',
                        use_amp_training=True)
        
        gan.configure(optimizer_generator,
                      optimizer_discriminator,
                      loss_generator,
                      loss_discriminator,
                      initialize_weights=initialize_weights_normal,
                      num_iter_discriminator=1,
                      num_accumulated=1, # Match fluvgan perfectly
                      fake_label_generator=1.,
                      real_label_discriminator=1.,
                      fake_label_discriminator=0.,
                      penalty_generator=None,
                      penalty_discriminator=penalty_discriminator)
        
        gan.train(dataset,
                  num_epochs=args.epochs,
                  batch_size=args.batch_size,
                  num_workers=4 * args.num_gpus if use_one_hot else 2 * args.num_gpus,
                  pin_memory=True,
                  drop_last=True,
                  checkpoint_step=1e12,
                  sampling_step=100,
                  sampling_size=3,
                  metrics=metrics,
                  metric_step=100,
                  validation_size=args.validation_size,
                  validation_batch_size=args.val_batch_size,
                  preload_validation=True,
                  resume_checkpoint_id=None)

        ################################################################################
        # Cleaning Checkpoints and Saving the Final `.pt` File
        
        checkpoint_dir_path = os.path.join(args.output_dir, f'{output_label}_Training_Checkpoints')
        checkpoint_paths = glob(os.path.join(checkpoint_dir_path, f'{output_label}_training_checkpoint_*'))
        
        if checkpoint_paths:
            checkpoint_paths = sorted(checkpoint_paths, key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
            last_checkpoint = checkpoint_paths[-1]

            new_filename = f"{model_name}_{dataset_name}_{encoding_tag}_epochs_{args.epochs}_bs_{args.batch_size}_run_{i + 1}.pt"
            final_checkpoint_path = os.path.join(args.output_dir, new_filename)

            shutil.copy(last_checkpoint, final_checkpoint_path)
            shutil.rmtree(checkpoint_dir_path)
            
            print(f"Saved final renamed model checkpoint to: {final_checkpoint_path}")

if __name__ == '__main__':
    main()