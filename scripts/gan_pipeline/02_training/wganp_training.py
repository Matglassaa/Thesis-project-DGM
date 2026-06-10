"""
WGAN-GP architecture training script:

example usage:
nohup python -u wganp_training.py \
    --run_name 10_epochs_3_classes_wgan_nexus_100_isbx \
    --data_file ~/data/datasets/training_dataset_nexus_20000_samples_ntg_0_67_chdepth_6_isbx_100/samples/facies/samples.h5 \
    --output_dir ~/data/outputs/UPD_20000_samples \
    --num_gpus 2 \
    --num_samples 10000 \
    --epochs 10 \
    --batch_size 64 \
    --val_batch_size 64 \
    --validation_size 0.2 \
    --one_hot_all False > training_wganp.out 2>&1 &
"""

import os
import re
import torch
import shutil
import warnings
import numpy as np
from glob import glob
import torch.nn as nn
import torch.optim as optim
from functools import partial

# Import Custom Modules
from dataloader import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks import resnet
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import WGANLoss, WGANGradientPenalty
from voxgan.models.metrics import MSSWD

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    config = parse_hybrid_args()

    np.random.seed(43)
    torch.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    
    config_save_path = os.path.join(run_dir, "config.json")
    save_config(config, config_save_path)
    print(f"Saved run configuration to: {config_save_path}")
    validate_dataset(config['data_file'])

    ################################################################################
    # Model configuration
    model_name = 'architecture_4_wgan_gp'
    nz = 100
    nl = (3, 5, 5)
    ngf = 64
    ndf = ngf
    max_factor = 16
    use_one_hot = not config['disable_one_hot']
    one_hot_all = config.get('one_hot_all', False)
    
    if use_one_hot:
        nc = 9 if one_hot_all else 3
        encoding_tag = "one_hot_all" if one_hot_all else "one_hot"
    else:
        nc = 1
        encoding_tag = "no_one_hot"
    
    # GENERATOR: Spectral Norm is okay here
    generator = partial(resnet.DeepGenerator3d,
                        nz=nz, 
                        ngf=ngf, 
                        nc=nc, 
                        nl=nl,
                        max_factor=max_factor, 
                        residual_weight=1., 
                        mode='nearest', 
                        kernel_size=3,
                        layer_normalization=nn.BatchNorm3d,
                        last_layer_normalization=nn.BatchNorm3d,
                        weight_normalization=nn.utils.parametrizations.spectral_norm,
                        activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                        last_activation=nn.Tanh,
                        use_double_conv=False, 
                        use_double_resblocks=False,
                        use_attention=False, 
                        skip_z=False, 
                        split_z=False)

    # CRITIC (Discriminator): 
    # 1. layer_normalization=None (BatchNorm ruins gradient penalty)
    # 2. weight_normalization=None (Let Gradient Penalty handle the Lipschitz constraint)
    discriminator = partial(resnet.DeepDiscriminator3d,
                            ndf=ndf, 
                            nc=nc, 
                            nl=nl,
                            max_factor=max_factor, 
                            residual_weight=1., 
                            kernel_size=3,
                            layer_normalization=nn.InstanceNorm3d, 
                            weight_normalization=None,
                            activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                            use_double_conv=False, 
                            use_double_resblocks=False,
                            use_attention=False)

    # WGAN-GP typically prefers Adam with beta1=0.0 and beta2=0.9
    optimizer_generator = partial(optim.Adam, lr=5e-5, betas=(0., 0.9))
    optimizer_discriminator = partial(optim.Adam, lr=5e-5, betas=(0., 0.9))
    
    loss_generator = WGANLoss()
    loss_discriminator = WGANLoss()
    penalty_discriminator = WGANGradientPenalty(gamma=10.)

    ################################################################################
    # Validation
    metric_params = dict(n_levels=3,
                         n_descriptors=1024,
                         descriptor_size=(4, 7, 7),
                         n_repeat=12,
                         n_proj=256,
                         padding_mode='circular', 
                         combine_levels=True, 
                         batch_size=config['val_batch_size'],
                         n_gpu=config['num_gpus'])

    if use_one_hot:
        metrics = [MSSWD(**metric_params, channel=c) for c in range(nc)]
    else:
        metrics = [MSSWD(**metric_params)]

    ################################################################################
    # Dataset Setup
    dataset_name = os.path.splitext(os.path.basename(config['data_file']))[0]
    
    dataset = partial(FaciesDataset,
                      h5_path=config['data_file'],
                      num_samples=config['num_samples'],
                      save_mapping_dir=run_dir,
                      use_one_hot=use_one_hot,
                      one_hot_all=one_hot_all,
                      dataset_name=dataset_name,
                      num_epochs=config['epochs'])

    ################################################################################
    # Training Loop
    num_training = 1 
    
    for i in range(num_training):
        output_label = f'{model_name}_{encoding_tag}_{i + 1}'
        
        # NOTE: If you experience NaN losses mid-training, WGAN-GP + AMP (Automatic Mixed Precision) 
        # can sometimes conflict due to double derivatives in the gradient penalty. 
        # If it crashes, change `use_amp_training` to False.
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=run_dir,
                        output_label=output_label,
                        verbose=2,
                        num_gpus=config['num_gpus'],
                        num_nodes=1,
                        distributed=False,
                        backend='nccl',
                        use_amp_training=False) 
        
        gan.configure(optimizer_generator,
                      optimizer_discriminator,
                      loss_generator,
                      loss_discriminator,
                      initialize_weights=initialize_weights_normal,
                      num_iter_discriminator=5,               
                      num_accumulated=1,
                      fake_label_generator=1.0,
                      real_label_discriminator=1.0,
                      fake_label_discriminator=-1.0,          
                      penalty_generator=None,
                      penalty_discriminator=penalty_discriminator) 
        
        gan.train(dataset,
                  num_epochs=config['epochs'],
                  batch_size=config['batch_size'],
                  num_workers=0 * config['num_gpus'],
                  pin_memory=True,
                  drop_last=True,
                  checkpoint_step=1e12,
                  sampling_step=100,
                  sampling_size=3,
                  metrics=metrics,
                  metric_step=100,
                  validation_size=config['validation_size'],
                  validation_batch_size=config['val_batch_size'],
                  preload_validation=True,
                  resume_checkpoint_id=None)

        ################################################################################
        # Saving and Post-Processing
        
        checkpoint_dir_path = os.path.join(run_dir, f'{output_label}_Training_Checkpoints')
        checkpoint_paths = glob(os.path.join(checkpoint_dir_path, f'{output_label}_training_checkpoint_*'))
        
        if checkpoint_paths:
            checkpoint_paths = sorted(checkpoint_paths, key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
            last_checkpoint = checkpoint_paths[-1]

            new_filename = f"{model_name}_{dataset_name}_{encoding_tag}_epochs_{config['epochs']}_bs_{config['batch_size']}_run_{i + 1}.pt"
            final_checkpoint_path = os.path.join(run_dir, new_filename)

            shutil.copy(last_checkpoint, final_checkpoint_path)
            shutil.rmtree(checkpoint_dir_path)
            
            print(f"Saved final renamed model checkpoint to: {final_checkpoint_path}\n")

            try:
                from visualize_and_generate import plot_losses, generate_realizations
                
                csv_path = os.path.join(run_dir, f"{output_label}_history.csv")
                plot_losses(csv_path, run_dir)
                
                generate_realizations(
                    ckpt_path=final_checkpoint_path, 
                    output_dir=run_dir, 
                    nc=nc, 
                    nl=nl,
                    num_realizations=100
                )
                print("Post-training visualization complete.")
            except Exception as e:
                print(f"Error during post-training visualization: {e}")

if __name__ == '__main__':
    main()