"""
WGAN-SN (MSG-GAN) architecture training script:

example usage:
nohup python -u wgans_training.py \
    --run_name 100_epochs_3_classes_mws-gan_nexus_100_isbx \
    --data_file ~/data/datasets/training_dataset_nexus_20000_samples_ntg_0_67_chdepth_6_isbx_100/samples/facies/samples.h5 \
    --output_dir ~/data/outputs/UPD_20000_samples \
    --num_gpus 2 \
    --num_samples 10000 \
    --epochs 100 \
    --batch_size 64 \
    --val_batch_size 64 \
    --validation_size 0.2 \
    --one_hot_all False > training_mwsgan.out 2>&1 &
"""

import os
import re
import torch
import shutil
import warnings
import numpy as np
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial

# Import Custom Modules
from dataloader import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import WGANLoss
from voxgan.models.metrics import MSSWD

warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NormalizedSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Swish has a max derivative of ~1.0998. 
        # Dividing by this guarantees the function is strictly 1-Lipschitz.
        return F.silu(x) / 1.0998
    
class GroupSort(nn.Module):
    def __init__(self, channels_per_group=2):
        super().__init__()
        self.c = channels_per_group

    def forward(self, x):
        # Sorts elements in groups of 2 along the channel dimension (dim=1)
        shape = x.shape
        x = x.view(shape[0], shape[1] // self.c, self.c, *shape[2:])
        x, _ = torch.sort(x, dim=2)
        return x.view(shape)

################################################################################
# Multi-Scale Generator (MSG-GAN)
################################################################################
class MSG_GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 1x1x1 convolution to generate the image at this scale
        self.to_rgb = nn.Conv3d(out_channels, img_channels, kernel_size=1)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out_act = nn.Tanh() # Required because real data is scaled to [-1, 1]

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        
        # Generate the multi-scale image output
        img_out = self.out_act(self.to_rgb(x))
        return x, img_out

class MultiScaleGenerator3D(nn.Module):
    def __init__(self, nz=100, nc=3):
        super().__init__()
        # EXPLICITLY SAVE ATTRIBUTES FOR VOXGAN COMPATIBILITY
        self.nz = nz  
        self.nc = nc  
        
        # Map latent noise to initial 4x16x16 block
        self.init_linear = nn.Linear(nz, 256 * 4 * 16 * 16) 
        
        self.block1 = MSG_GeneratorBlock(256, 128, nc) # Outputs 8x32x32
        self.block2 = MSG_GeneratorBlock(128, 64, nc)  # Outputs 16x64x64
        self.block3 = MSG_GeneratorBlock(64, 32, nc)   # Outputs 32x128x128
        
    def forward(self, z):
        # --- THE GENERATOR FIX ---
        # VoxGAN compatibility: unwrap the dictionary to get the raw latent tensor
        if isinstance(z, dict):
            z = z['data']
            
        # Now z is guaranteed to be a PyTorch tensor, and .view() will work
        z = z.view(z.size(0), -1) 
        x = self.init_linear(z).view(-1, 256, 4, 16, 16)
        
        x, img1 = self.block1(x)
        x, img2 = self.block2(x)
        x, final_img = self.block3(x)
        
        # The VoxGAN compatibility Hack
        return {
            'data': final_img,       # VoxGAN metrics use this
            'ms_fakes': [img1, img2] # Our Discriminator uses this
        }

################################################################################
# Multi-Scale Critic (MSG-GAN)
################################################################################
class MinibatchStdev3D(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_32 = x.to(torch.float32)
        batch_size, _, d, h, w = x_32.shape
        mean = torch.mean(x_32, dim=0, keepdim=True)
        sq_diff = torch.square(x_32 - mean)
        variance = torch.mean(sq_diff, dim=0, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        mean_std = torch.mean(std)
        target_shape = (batch_size, 1, d, h, w)
        expanded_std = mean_std.expand(target_shape).to(orig_dtype)
        return torch.cat([x, expanded_std], dim=1)

class MSG_DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mstd = MinibatchStdev3D()
        # +1 channel to account for MinibatchStdev
        self.conv1 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(in_channels + 1, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        self.down = nn.AvgPool3d(2)
        self.act = GroupSort()

    def forward(self, x, img):
        # 1. Combine previous block features with the current scale image
        if x is None:
            h = img
        else:
            h = torch.cat([x, img], dim=1)
            
        # 2. Apply Minibatch StdDev (+1 channel)
        h = self.mstd(h)
        
        # 3. Convolutions & Activation
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        
        # 4. Downsample to the next scale
        h = self.down(h)
        return h

class MultiScaleDiscriminator3D(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        # Top-Down evaluation
        self.block1 = MSG_DiscriminatorBlock(nc, 32)
        self.block2 = MSG_DiscriminatorBlock(32 + nc, 64)
        self.block3 = MSG_DiscriminatorBlock(64 + nc, 128)
        self.flatten = nn.Flatten()
        self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(128 * 4 * 16 * 16, 1))

    def forward(self, input_dict):
        # Fallback for metric validation which passes raw tensors instead of dicts
        if isinstance(input_dict, torch.Tensor):
            final_img = input_dict
            ms_fakes = None
        else:
            final_img = input_dict['data']
            ms_fakes = input_dict.get('ms_fakes', None)
        
        if ms_fakes is not None:
            img_8x32 = ms_fakes[0]
            img_16x64 = ms_fakes[1]
        else:
            # Generate Real multi-scale images on the fly via AvgPool
            img_16x64 = F.avg_pool3d(final_img, 2)
            img_8x32 = F.avg_pool3d(final_img, 4)
            
        # Top-Down evaluation: Highest resolution first
        x = self.block1(None, final_img)     # Output features: 16x64x64
        x = self.block2(x, img_16x64)        # Output features: 8x32x32
        x = self.block3(x, img_8x32)         # Output features: 4x16x16
        
        return {'data': self.fc(self.flatten(x))}

class ChannelAwareMSSWD(MSSWD):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.custom_name = name

    def __str__(self):
        return self.custom_name


def generate_realizations(ckpt_path, output_dir, nc=3, nl=(3,5,5), num_realizations=100):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at '{ckpt_path}'. Skipping generation.")
        return
        
    output_dir = os.path.join(output_dir, 'realizations')
    os.makedirs(output_dir, exist_ok=True)

    nz = 100

    print("Building Multi-Scale Generator (MSG-GAN)...")
    # 1. Instantiate the new MSG-GAN generator instead of the ResNet
    gen_layer = MultiScaleGenerator3D(nz=nz, nc=nc)

    print(f"Loading Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    gen_state = checkpoint['generator'] if 'generator' in checkpoint else checkpoint
    if 'state_dict' in gen_state:
        gen_state = gen_state['state_dict']

    # 2. VoxGAN saves models wrapped in DataParallel, so we wrap it here before loading
    gen_layer = nn.DataParallel(gen_layer)
    
    # Load the weights (I added a print statement here so you can catch shape mismatches)
    missing, unexpected = gen_layer.load_state_dict(gen_state, strict=False)
    if missing or unexpected:
        print(f"Warning during loading:\nMissing keys: {missing}\nUnexpected keys: {unexpected}")
        
    gen_layer = gen_layer.module # Unwrap from DataParallel
    gen_layer.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_layer = gen_layer.to(device)
    print("Success! Weights are perfectly aligned.")

    print(f"Generating {num_realizations} 3D River Block Realizations...")
    for i in range(num_realizations):
        with torch.no_grad():
            # 3. Generate latent vector (MSG-GAN handles the flattening internally now)
            z = torch.randn(1, nz).to(device)
            
            # Pass through the MSG generator
            output = gen_layer(z)
            
            # 4. Extract the highest resolution output using our 'data' dictionary hack
            sample = output['data'].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
            
            if nc > 1:
                # Convert one-hot (Channels, Depth, Height, Width) to categorical
                volume = np.argmax(sample[0], axis=0).astype(np.uint8)
            else:
                volume = sample[0, 0, :, :, :]

            out_path = os.path.join(output_dir, f"realization_{i+1:02d}.npy")
            np.save(out_path, volume)
            
            if (i + 1) % 10 == 0 or (i + 1) == num_realizations:
                print(f"Generated realization {i+1}/{num_realizations} -> Data shape= {volume.shape}")

    print("Generation complete!")

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

    model_name = 'architecture_msg_wgan_sn'
    nz = 100
    nl = (3, 5, 5) # Used for visualizations/metadata
    use_one_hot = not config['disable_one_hot']
    one_hot_all = config.get('one_hot_all', False)
    
    if use_one_hot:
        nc = 9 if one_hot_all else 3
        encoding_tag = "one_hot_all" if one_hot_all else "one_hot"
    else:
        nc = 1
        encoding_tag = "no_one_hot"
    
    # Initialize our custom MSG classes using partial to pass constructor args
    generator = partial(MultiScaleGenerator3D, nz=nz, nc=nc)
    discriminator = partial(MultiScaleDiscriminator3D, nc=nc)

    optimizer_generator = partial(optim.Adam, lr=1e-5, betas=(0., 0.9))
    optimizer_discriminator = partial(optim.Adam, lr=1e-5, betas=(0., 0.9))
    
    loss_generator = WGANLoss()
    loss_discriminator = WGANLoss()

    metric_params = dict(n_levels=3,
                         n_descriptors=256,
                         descriptor_size=(4, 7, 7),
                         n_repeat=12,
                         n_proj=56,
                         padding_mode='circular', 
                         combine_levels=True, 
                         batch_size=config['val_batch_size'],
                         n_gpu=config['num_gpus'])

    if use_one_hot:
        metrics = []
        for c in range(nc):
            metric_name = f"MS-SWD_Ch{c}"
            ms_swd = ChannelAwareMSSWD(name=metric_name, channel=c, **metric_params)
            metrics.append(ms_swd)
    else:
        ms_swd = ChannelAwareMSSWD(name="MS-SWD", **metric_params)
        metrics = [ms_swd]

    dataset_name = os.path.splitext(os.path.basename(config['data_file']))[0]
    
    dataset = partial(FaciesDataset,
                      h5_path=config['data_file'],
                      num_samples=config['num_samples'],
                      save_mapping_dir=run_dir,
                      use_one_hot=use_one_hot,
                      one_hot_all=one_hot_all,
                      dataset_name=dataset_name,
                      num_epochs=config['epochs'])

    num_training = 1 
    
    for i in range(num_training):
        output_label = f'{model_name}_{encoding_tag}_{i + 1}'
        
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=run_dir,
                        output_label=output_label,
                        verbose=2,
                        num_gpus=config['num_gpus'],
                        num_nodes=1,
                        distributed=False,
                        backend='nccl',
                        use_amp_training=False) # Keep False for FP32 stability!
        
        gan.configure(optimizer_generator,
                      optimizer_discriminator,
                      loss_generator,
                      loss_discriminator,
                      initialize_weights=initialize_weights_normal,
                      num_iter_discriminator=5,               
                      num_accumulated=int(64 / config['batch_size']),
                      fake_label_generator=-1.0,       
                      real_label_discriminator=-1.0,   
                      fake_label_discriminator=1.0 ,        
                      penalty_generator=None,
                      penalty_discriminator=None) 
        
        gan.train(dataset,
                  num_epochs=config['epochs'],
                  batch_size=config['batch_size'],
                  num_workers=0 * config['num_gpus'],
                  pin_memory=True,
                  drop_last=True,
                  checkpoint_step=1e12,
                  sampling_step=1000,
                  sampling_size=3,
                  metrics=metrics,
                  metric_step=20,
                  validation_size=config['validation_size'],
                  validation_batch_size=config['val_batch_size'],
                  preload_validation=True,
                  resume_checkpoint_id=None)

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
                from visualize_and_generate import plot_losses
                
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