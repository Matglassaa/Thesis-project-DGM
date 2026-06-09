################################################################################
# Imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import os
from pathlib import Path
import shutil
from functools import partial
import numpy as np
from glob import glob
from torchinfo import summary

import torch
import torch.nn as nn
import torch.optim as optim

# Import Custom Modules
from scripts.old_scripts.old_gan_scripts.GAN.dataloader_old import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks import resnet
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import R1Regularization
from voxgan.models.metrics import MSSWD, LoS

def main():
    config = parse_hybrid_args()

    ################################################################################
    # Setting
    np.random.seed(43)
    torch.manual_seed(43)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup run directory
    run_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration to the run directory
    config_save_path = os.path.join(run_dir, "config.json")
    save_config(config, config_save_path)
    print(f"Saved run configuration to: {config_save_path}")

    # Validate the dataset before training
    validate_dataset(config['data_file'])

    ################################################################################
    # Model configuration
    model_name = 'architecture_4_dcgan'
    nz = 100
    nl = (3, 5, 5)
    
    use_one_hot = not config['disable_one_hot']
    one_hot_all = config.get('one_hot_all', False) # Assuming you add this to your args
    if use_one_hot:
        nc = 9 if one_hot_all else 3
    else:
        nc = 1
        
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
    
    gen_model = generator()
    disc_model = discriminator()

    # 2. Print summaries to console and optionally save to text
    print("\n" + "="*50)
    print("GENERATOR ARCHITECTURE:")
    gen_summary = summary(gen_model, input_size=(1, nz), verbose=1)

    print("\n" + "="*50)
    print("DISCRIMINATOR ARCHITECTURE:")
    # Adjust spatial dimensions to match your dataset (e.g., 64x64x64 or 128x128x128)
    disc_summary = summary(disc_model, input_size=(1, nc, 128, 128, 128), verbose=1)

    # Save to a text file in your run_dir
    with open(os.path.join(run_dir, "model_architectures.txt"), "w") as f:
        f.write("GENERATOR:\n")
        f.write(str(gen_summary))
        f.write("\n\nDISCRIMINATOR:\n")
        f.write(str(disc_summary))

    optimizer_generator = partial(optim.Adam, lr=5e-5, betas=(0., 0.99))
    optimizer_discriminator = partial(optim.Adam, lr=5e-5, betas=(0., 0.99))
    
    loss_generator = nn.BCEWithLogitsLoss()
    loss_discriminator = nn.BCEWithLogitsLoss()
    
    penalty_discriminator = R1Regularization(gamma=10., num_iter=16, use_amp=True)      # Remove entirely?

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

    #### COMMENT: increase n_descriptors in the MSSWD metric -> larger 3D blocks so more information to be processed!n
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
        metrics = []
        
        # Dynamically create an MSSWD metric for every single facies channel
        for channel_idx in range(nc):
            ms_swd = MSSWD(**metric_params, channel=channel_idx)
            metrics.append(ms_swd)
            
        # Add the Line of Sight (LoS) metric (leaving it on channel 0 as in your original code)
        los = LoS(channel=0, batch_size=config['val_batch_size'], n_gpu=config['num_gpus'])
        metrics.append(los)
        
    else:
        ms_swd = MSSWD(**metric_params)
        los = LoS(channel=0, batch_size=config['val_batch_size'], n_gpu=config['num_gpus'])
        metrics = [ms_swd, los]

    ################################################################################
    # Dataset
    dataset_name = os.path.splitext(os.path.basename(config['data_file']))[0]
    
    dataset = partial(FaciesDataset,
                      h5_path=config['data_file'],
                      num_samples = config['num_samples'],
                      save_mapping_dir=run_dir,
                      use_one_hot=use_one_hot,
                      one_hot_all=one_hot_all,  
                      dataset_name=dataset_name,
                      num_epochs=config['epochs'])

    ################################################################################
    # Training
    num_training = 1 
    
    for i in range(num_training):
        output_label = f'fluvgan_1_training_1_{model_name}_{encoding_tag}_{i + 1}'
        
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=run_dir,
                        output_label=output_label,
                        verbose=2,
                        num_gpus=config['num_gpus'],
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
                      num_accumulated=1, 
                      fake_label_generator=1.,
                      real_label_discriminator=1.0,
                      fake_label_discriminator=0.0,
                      penalty_generator=None
                      )
                      #penalty_discriminator=penalty_discriminator)
        
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
        # Cleaning Checkpoints and Saving the Final `.pt` File
        
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

            ################################################################################
            # Visualization and Generation (Post-training)
            print("-" * 35)
            print("Running post-training visualization and generation...")
            try:
                from scripts.old_scripts.old_gan_scripts.GAN.visualize_and_generate_old import plot_losses, generate_realizations
                
                # Plot losses
                csv_path = os.path.join(run_dir, f"{output_label}_history.csv")
                plot_losses(csv_path, run_dir)
                
                # Generate realizations
                generate_realizations(nc,final_checkpoint_path, run_dir, num_realizations=10)
                print("Post-training visualization complete.")
            except ImportError as e:
                print(f"Could not import visualization modules: {e}")
            except Exception as e:
                print(f"Error during post-training visualization: {e}")

if __name__ == '__main__':
     main()

# # EXAMPLE RUN: nohup python -u gan_training.py --run_name test_run_01 
# #               --data_file ~/data/training_dataset_upper_plain_delta_128/training_datset_upper_plane_delta.h5 
# #               --output_dir ~/data/outputs_upper_plain_delta_128   
# #               --num_gpus 2 --epochs 5 --batch_size 8 --val_batch_size 8 --validation_size 0.1 --disable_one_hot > training.out 2>&1 &
# # nohup python -u gan_training.py --run_name RUN_2000_samples_128xy_dataset_50_epochs --data_file ~/data/datasets/training_dataset_upper_plain_delta_128/training_datset_upper_plane_delta.h5 --output_dir ~/data/outputs/UPD_2000_samples_128xy_seed_43 