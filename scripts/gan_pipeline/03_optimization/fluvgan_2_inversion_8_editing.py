"""
==================================
GAN inversion using pivotal tuning
==================================

Architecture DCGAN (4)
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator
             + R1 regularization

Adapted for cluster execution and Excel-based well data.

Example Run:
nohup python fluvgan_2_inversion_8_editing.py \
    --model_path ~/data/outputs/UPD_20000_samples_128xy_seed_43/RUN_10000_of_20000_samples_10_epochs_bs_64_val_size_010_no_disc_penalty/architecture_4_dcgan_training_datset_upper_plane_delta_20000_one_hot_epochs_100_bs_64_run_1.pt \
    --optimized_z_path ~/data/outputs/post_optimization_results/RUN_10000_of_20000_samples_10_epochs_bs_64_val_size_010_no_disc_penalty/optimized_z.npy \
    --well_data_path ~/data/datasets/well_data/Well_data.xlsx \
    --output_dir ~/data/outputs/post_inversion_results/RUN_10000_of_20000_samples_10_epochs_bs_64_val_size_010_no_disc_penalty/ \
    --batch_size 64 --steps 1000 > editing.out 2>&1 &
"""

import os
import re
import argparse
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from voxgan.networks import resnet

################################################################################
# Functions

class LPIPSLoss(_Loss):
    """
    LPIPS loss with a 3D discriminator.
    """
    def __init__(self, discriminator, reduction='mean', eps=1e-12):
        super(LPIPSLoss, self).__init__(None, None, None)
        self.discriminator = discriminator
        self.reduction = torch.mean if reduction == 'mean' else (torch.sum if reduction == 'sum' else lambda x: x)
        self.eps = eps

    def forward(self, input, target):
        loss = 0.
        disc = self.discriminator.module if isinstance(self.discriminator, nn.DataParallel) else self.discriminator
        
        curr_input = input
        curr_target = target
        
        for i in range(len(disc.main) - 1):
            curr_input = disc.main[i](curr_input)
            curr_target = disc.main[i](curr_target)
            
            # Normalize ONLY for the distance calculation
            norm_input = F.normalize(curr_input, eps=self.eps, dim=1)
            norm_target = F.normalize(curr_target, eps=self.eps, dim=1)
            
            loss += torch.mean(torch.sum((norm_input - norm_target)**2, 1), (1, 2, 3))

        return self.reduction(loss)

def map_facies(val):
    """Maps raw facies values to 3 classes."""
    if 1 <= val <= 3: return 0
    if 4 <= val <= 7: return 1
    if 8 <= val <= 12: return 2
    return 0

################################################################################
# Main Execution

def main():
    parser = argparse.ArgumentParser(description="GAN inversion using pivotal tuning (editing)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument('--optimized_z_path', type=str, required=True, help="Path to optimized_z.npy")
    parser.add_argument('--well_data_path', type=str, required=True, help="Path to Well_data.xlsx")
    parser.add_argument('--output_dir', type=str, default='../outputs/editing', help="Output directory")
    parser.add_argument('--nz', type=int, default=100, help="Latent vector size")
    parser.add_argument('--nc', type=int, default=3, help="Number of facies channels")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for tuning")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate for generator tuning")
    parser.add_argument('--steps', type=int, default=1000, help="Number of tuning steps")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Automate output directory structure
    model_parent = Path(args.model_path).parent.name
    final_output_dir = os.path.join(args.output_dir, model_parent)
    realizations_dir = os.path.join(final_output_dir, 'realizations')
    os.makedirs(realizations_dir, exist_ok=True)

    ################################################################################
    # Model Setup
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    nl = (3, 5, 5)
    
    generator = resnet.DeepGenerator3d(nz=args.nz,
                                       ngf=64,
                                       nc=args.nc,
                                       nl=nl,
                                       max_factor=16,
                                       residual_weight=1.,
                                       mode='nearest',
                                       kernel_size=3,
                                       layer_normalization=nn.BatchNorm3d,
                                       last_layer_normalization=nn.BatchNorm3d,
                                       weight_normalization=nn.utils.parametrizations.spectral_norm,
                                       activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=False),
                                       last_activation=nn.Tanh,
                                       use_double_conv=False,
                                       use_double_resblocks=False,
                                       use_attention=False,
                                       skip_z=False,
                                       split_z=False)
    
    discriminator = resnet.DeepDiscriminator3d(ndf=64,
                                               nc=args.nc,
                                               nl=nl,
                                               max_factor=16,
                                               residual_weight=1.,
                                               kernel_size=3,
                                               layer_normalization=None,
                                               weight_normalization=nn.utils.parametrizations.spectral_norm,
                                               activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=False),
                                               use_double_conv=True, # Matches training
                                               use_double_resblocks=False,
                                               use_attention=False)

    # Clean state dicts
    gen_state = {k.replace('module.', ''): v for k, v in checkpoint['generator'].items()}
    disc_state = {k.replace('module.', ''): v for k, v in checkpoint['discriminator'].items()}
    
    generator.load_state_dict(gen_state)
    discriminator.load_state_dict(disc_state)
    
    generator.to(device).eval()
    discriminator.to(device).eval()
    
    for p in generator.parameters(): p.requires_grad = False
    for p in discriminator.parameters(): p.requires_grad = False

    ################################################################################
    # Data Processing
    
    print("Loading optimized latent vectors...")
    p_z = np.load(args.optimized_z_path)
    p_z = torch.tensor(p_z, device=device).float()
    n_samples = p_z.shape[0]

    print("Loading well data...")
    ORIGIN_E = 84337.0
    ORIGIN_N = 445750.0
    SPACING = 20.0
    
    tabs = ['DEL-GT-01', 'DEL-GT-02-S2']
    all_indices = []
    all_values = []
    
    for tab in tabs:
        df = pd.read_excel(args.well_data_path, sheet_name=tab)
        df_32 = df.head(32)
        facies_col = 'Facies ' if 'Facies ' in df.columns else 'Facies'
        
        for i, row in df_32.iterrows():
            z = i
            y = int(np.round((row['GRID N'] - ORIGIN_N) / SPACING))
            x = int(np.round((row['GRID E'] - ORIGIN_E) / SPACING))
            if 0 <= x < 128 and 0 <= y < 128:
                all_indices.append([z, y, x])
                all_values.append(map_facies(row[facies_col]))

    X = torch.tensor(all_indices).T # (3, N)
    y_vals = torch.tensor(all_values)
    
    # One-hot targets in [-1, 1] range
    one_hot_targets = torch.full((len(all_values), args.nc), -1.0, device=device)
    for i, v in enumerate(all_values):
        one_hot_targets[i, int(v)] = 1.0
    one_hot_targets = one_hot_targets.T # (3, N)

    ################################################################################
    # Pivotal Tuning
    
    print(f"Starting pivotal tuning for {args.steps} steps...")
    tuned_generator = copy.deepcopy(generator)
    tuned_generator.requires_grad_(True)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for tuning!")
        tuned_generator = nn.DataParallel(tuned_generator)
        # lpips needs discriminator, wrap it too if multiple GPUs
        dist_discriminator = nn.DataParallel(discriminator)
    else:
        dist_discriminator = discriminator

    optimizer = torch.optim.Adam(tuned_generator.parameters(), lr=args.lr)
    
    loss_fn_tuning = nn.L1Loss()        # Use L1 or MSE for tuning against [-1, 1] targets
    loss_fn_reg_l2 = nn.MSELoss()       # Use true L2 loss for comparing the [-1, 1] generator outputs 
    loss_fn_reg_lpips = LPIPSLoss(dist_discriminator)

    history = {'loss': []}
    pbar = tqdm(range(args.steps))
    
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.
        
        # Batch through the optimized z samples
        for i in range(0, n_samples, args.batch_size):
            s = slice(i, i + args.batch_size)
            curr_batch_size = p_z[s].shape[0]
            
            # 1. Tuning Loss: Match optimized z to well data
            samples = tuned_generator(p_z[s].view(curr_batch_size, args.nz, 1, 1, 1))
            # extracted: (B, 3, N)
            extracted = samples[:, :, X[0], X[1], X[2]]
            loss_tuning = loss_fn_tuning(extracted, one_hot_targets.expand(curr_batch_size, -1, -1))
            
            # 2. Regularization Loss: Keep the model near original for random z
            # Sample z near the pivot
            z_rand = torch.randn(curr_batch_size, args.nz, device=device)
            # Distance controlled interpolation (Roich et al. 2021 style)
            z_pivot = p_z[s]
            dist = torch.linalg.norm(z_rand - z_pivot, dim=1, keepdim=True)
            z_interp = z_pivot + 30. * (z_rand - z_pivot) / dist
            
            with torch.no_grad():
                samples_orig = generator(z_interp.view(curr_batch_size, args.nz, 1, 1, 1))
            
            samples_tuned = tuned_generator(z_interp.view(curr_batch_size, args.nz, 1, 1, 1))
            
            loss_reg_l2 = loss_fn_reg_l2(samples_tuned, samples_orig)
            loss_reg_lpips = loss_fn_reg_lpips(samples_tuned, samples_orig)
            
            # weighting from Roich et al. (2021) adapted for this setup
            # # Increase LPIPS weight from 0.1 to 0.5 (or even 1.0) to hold the structural geology together
            # loss = loss_tuning + 0.5 * loss_reg_lpips + 0.1 * loss_reg_l2
            loss = loss_tuning + 0.1 * (loss_reg_lpips + 1.0 * loss_reg_l2)
            
            loss = loss * (curr_batch_size / n_samples)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        history['loss'].append(total_loss)
        if step % 10 == 0:
            pbar.set_description(f"Loss: {total_loss:.4f}")

    ################################################################################
    # Saving Results
    
    print(f"Saving results to {final_output_dir}...")
    pd.DataFrame(history).to_csv(os.path.join(final_output_dir, 'tuning_history.csv'), index=False)
    
    # Save tuned model
    save_model = tuned_generator.module if isinstance(tuned_generator, nn.DataParallel) else tuned_generator
    torch.save(save_model.state_dict(), os.path.join(final_output_dir, 'tuned_generator.pt'))
    
    # Generate and save final realizations
    with torch.no_grad():
        save_model.eval()
        for i in range(n_samples):
            _z = p_z[i].view(1, args.nz, 1, 1, 1)
            sample = save_model(_z)
            sample_np = (0.5 * sample + 0.5).cpu().numpy()[0]
            np.save(os.path.join(realizations_dir, f'realization_{i+1}.npy'), sample_np)

    print("Editing complete!")

if __name__ == '__main__':
    main()
