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

Roich, D., Mokady, R., Bermano, A. H., & Cohen-Or, D. (2021)
    Pivotal tuning for latent-based editing of real images 	
    https://doi.org/10.48550/arXiv.2106.05744

"""

# Author: Guillaume Rongier
# License: MIT


################################################################################
# Imports

import re
from pathlib import Path
from functools import partial
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

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

    Parameters
    ----------
    discriminator : nn.Module
        The trained discriminator of a GAN for 3D images.
    reduction : str, default='mean'
        Reduction to apply to the output:
            - 'none': no reduction will be applied.
            - 'mean': the sum of the output will be divided by the number of
               elements in the output.
            - 'sum': the output will be summed.
    eps : float, default=1e-12
        Small number used in tensor normalization to avoid division by zero.

    References
    ----------
    Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018)
        The unreasonable effectiveness of deep features as a perceptual metric
        https://arxiv.org/abs/1801.03924
    Pan, X., Zhan, X., Dai, B., Lin, D., Loy, C. C., & Luo, P. (2021)
        Exploiting deep generative prior for versatile image restoration and manipulation
        https://doi.org/10.1109/TPAMI.2021.3115428
    """
    def __init__(self, discriminator, reduction='mean', eps=1e-12):

        super(LPIPSLoss, self).__init__(None, None, None)

        self.discriminator = discriminator
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            self.reduction = lambda x: x
        self.eps = eps

    def forward(self, input, target):

        loss = 0.
        for i in range(len(self.discriminator.main) - 1):
            input = self.discriminator.main[i](input)
            input = F.normalize(input, eps=self.eps)
            target = self.discriminator.main[i](target)
            target = F.normalize(target, eps=self.eps)
            loss += torch.mean(torch.sum((input - target)**2, 1), (1, 2, 3))

        return self.reduction(loss)


################################################################################
# Paths

model_dir_path = Path('../outputs')
data_dir_path = Path('../data')
output_dir_path = Path('../outputs')


################################################################################
# Setting

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


torch.manual_seed(42)


################################################################################
# Model

for i_model in ['1', '2', '3']:
    print('Model:', i_model)

    model_name = 'fluvgan_1_training_6_latent-size_128_' + i_model + '_training_checkpoint_*'

    checkpoint_paths = sorted(list(model_dir_path.glob(model_name)), key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
    checkpoint = torch.load(checkpoint_paths[-1], device)

    generator = resnet.DeepGenerator3d(nz=128,
                                       ngf=64,
                                       nc=2,
                                       nl=(2, 5, 5),
                                       max_factor=16,
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
    generator = nn.DataParallel(generator)
    generator.load_state_dict(checkpoint['generator'])
    generator = generator.module
    generator.eval()
    generator.requires_grad_ = False

    discriminator = resnet.DeepDiscriminator3d(ndf=64,
                                               nc=2,
                                               nl=(2, 5, 5),
                                               max_factor=16,
                                               residual_weight=1.,
                                               kernel_size=3,
                                               layer_normalization=None,
                                               weight_normalization=nn.utils.parametrizations.spectral_norm,
                                               activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                               use_double_conv=False,
                                               use_double_resblocks=False,
                                               use_attention=False)
    discriminator = nn.DataParallel(discriminator)
    discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator = discriminator.module
    discriminator.requires_grad_ = False


################################################################################
# Data

    for i_sample in [20001, 20002, 20003]:
        print('... Data sample:', i_sample)

        ground_truth = xr.load_dataset(data_dir_path/'fluvgan_2_inversion_ground-truth.nc')
        ground_truth = torch.tensor(ground_truth[str(i_sample)].to_numpy())
        ground_truth[1] = (ground_truth[1] - ground_truth[1].min())/(ground_truth[1].max() - ground_truth[1].min())

        for n_wells in [4, 8, 20]:
            print('... ... Number of wells:', n_wells)

            data = pd.read_csv(data_dir_path/'fluvgan_2_inversion_well-data.csv')

            is_valid = (data['Sample'] == i_sample) & (data['Well'] <= n_wells)
            X = torch.tensor(data.loc[is_valid, ['K', 'J', 'I']].to_numpy()).int().T
            y = torch.tensor(data.loc[is_valid, 'Fraction'].to_numpy()).float()


################################################################################
# Inference

            batch_size = 32

            tuned_generator = copy.deepcopy(generator)
            tuned_generator.requires_grad_ = True

            file_name = 'fluvgan_2_inversion_3_prior-art_1_' + str(i_sample) + '_' + str(n_wells) + '_' + i_model + '_z.npy'
            p = np.load(output_dir_path/file_name)
            p = torch.tensor(p)
            n_samples = p.shape[0]

            optimizer = torch.optim.Adam(tuned_generator.parameters(), lr=3e-4)
            loss_fn_tuning = nn.MSELoss(reduction='none')
            loss_fn_reg_l2 = nn.MSELoss(reduction='none')
            loss_fn_reg_lpips = LPIPSLoss(discriminator, reduction='none')

            history = {
                'loss': [],
            }
            for step in tqdm(range(1000)):

                optimizer.zero_grad()
                _loss = 0.
                for i in range(0, n_samples, batch_size):

                    s = slice(i, i + batch_size)
                    samples = tuned_generator(p[s].view(*p[s].shape, 1, 1, 1))
                    loss_tuning = loss_fn_tuning(samples[:, 0, X[0], X[1], X[2]], 2.*y - 1.).mean(1).sum()/n_samples

                    z = torch.randn(min(batch_size, n_samples - i), generator.nz)
                    z = p[s] + 30.*(z - p[s])/torch.linalg.norm(z - p[s], dim=1, keepdim=True)
                    tuned_samples = tuned_generator(z.view(*z.shape, 1, 1, 1))
                    samples = generator(z.view(*z.shape, 1, 1, 1))
                    loss_reg_l2 = loss_fn_reg_l2(samples, tuned_samples).mean((1, 2, 3, 4)).sum()/n_samples
                    loss_reg_lpips = loss_fn_reg_lpips(samples, tuned_samples).sum()/n_samples

                    loss = loss_tuning + 0.1*(loss_reg_lpips + 1.*loss_reg_l2)
                    _loss += loss
                    loss.backward()

                optimizer.step()
                history['loss'].append(_loss.item())
        
            history = pd.DataFrame(history)


################################################################################
# Error

            batch_size = 100

            loss_fn_mae = nn.L1Loss(reduction='none')
            loss_fn_mse = nn.MSELoss(reduction='none')

            error = {'MAE': np.empty(n_samples),
                     'RMSE': np.empty(n_samples),
                     'MAE_GT_FRA': np.empty(n_samples),
                     'RMSE_GT_FRA': np.empty(n_samples),
                     'MAE_GT_AGE': np.empty(n_samples),
                     'RMSE_GT_AGE': np.empty(n_samples)}
            with torch.no_grad():
                for i in tqdm(range(0, n_samples, batch_size)):
                    s = slice(i, i + batch_size)

                    samples = 0.5*tuned_generator(p[s].view(p[s].shape + (1, 1, 1))) + 0.5

                    error['MAE'][s] = loss_fn_mae(y, samples[:, 0, X[0], X[1], X[2]]).mean(1).cpu().numpy()
                    error['RMSE'][s] = torch.sqrt(loss_fn_mse(y, samples[:, 0, X[0], X[1], X[2]]).mean(1)).cpu().numpy()

                    error['MAE_GT_FRA'][s] = loss_fn_mae(ground_truth[0], samples[:, 0]).mean((1, 2, 3)).cpu().numpy()
                    error['RMSE_GT_FRA'][s] = torch.sqrt(loss_fn_mse(ground_truth[0], samples[:, 0]).mean((1, 2, 3))).cpu().numpy()
                    error['MAE_GT_AGE'][s] = loss_fn_mae(ground_truth[1], samples[:, 1]).mean((1, 2, 3)).cpu().numpy()
                    error['RMSE_GT_AGE'][s] = torch.sqrt(loss_fn_mse(ground_truth[1], samples[:, 1]).mean((1, 2, 3))).cpu().numpy()

            error = pd.DataFrame(error)

            print('... ... ... MAE:', error['MAE'].mean(), '| RMSE:', error['RMSE'].mean())


################################################################################
# Saving

            label = 'fluvgan_2_inversion_8_editing_' + str(i_sample) + '_' + str(n_wells) + '_' + i_model + '_'

            history.to_csv(output_dir_path/(label + 'history.csv'), index=False)
            np.save(output_dir_path/(label + 'z.npy'), p.cpu().detach().numpy())
            error.to_csv(output_dir_path/(label + 'error.csv'), index=False)
            torch.save(tuned_generator.state_dict(), output_dir_path/(label + 'tunedgenerator.pt'))
