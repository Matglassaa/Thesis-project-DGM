"""
==============================================
GAN inversion using latent-vector optimization
==============================================

Architecture DCGAN (4)
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator
             + R1 regularization

Dupont, E., Zhang, T., Tilke, P., Liang, L., & Bailey, W. (2018)
    Generating realistic geology conditioned on physical measurements with generative adversarial networks
    https://doi.org/10.48550/arXiv.1802.03065

"""

# Author: Guillaume Rongier
# License: MIT


################################################################################
# Imports

import re
from pathlib import Path
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from voxgan.networks import resnet


################################################################################
# Functions

class ContextLoss(_Loss):
    """
    Context loss with an expanded mask for 3D images.

    Parameters
    ----------
    data_indices : array-like of shape (3, n)
        Indices of the target data in the 3D image.
    data_values : array-like of shape (n,)
        Values of the target data.
    threshold : float
        Distance threshold after which the expanded data are ignored, in the
        same unit as `spacing`.
    shape : array-like of shape (3,)
        Shape of the 3D image.
    spacing : float or array-like of shape (3,), default=None
        Cell size of the 3D image.
    p : float, default=1.
        Exponent of the distance. If 1, the absolute function is used instead.
    reduction : str, default='sum'
        Reduction to apply to the output:
            - 'none': no reduction will be applied.
            - 'mean': the sum of the output will be divided by the number of
               elements in the output.
            - 'sum': the output will be summed.
    device : Device, default=None
        PyTorch device to move the tensor to.

    References
    ----------
    Dupont, E., Zhang, T., Tilke, P., Liang, L., & Bailey, W. (2018)
        Generating realistic geology conditioned on physical measurements with generative adversarial networks
        https://arxiv.org/abs/1802.03065
    """
    def __init__(self,
                 data_indices,
                 data_values,
                 threshold,
                 shape,
                 spacing=None,
                 p=1,
                 reduction='sum',
                 device=None):

        super(ContextLoss, self).__init__(None, None, reduction)

        input = torch.ones(shape, dtype=int)
        input[data_indices[0], data_indices[1], data_indices[2]] = 0
        distance, indices = distance_transform_edt(input.cpu(), sampling=spacing, return_indices=True)

        self._target = torch.empty(shape, device=device)
        self._target[data_indices[0], data_indices[1], data_indices[2]] = data_values
        self._target = self._target[indices[0], indices[1], indices[2]]

        self._mask = 1./np.sqrt(distance + 1.)
        self._mask[distance > threshold] = 0.
        self._mask = torch.tensor(self._mask).to(device)

        if p == 1:
            self._p = torch.abs
        else:
            self._p = partial(torch.pow, exponent=p)

        if reduction == 'mean':
            self.reduction = partial(torch.mean, dim=(1, 2, 3))
        elif reduction == 'sum':
            self.reduction = partial(torch.sum, dim=(1, 2, 3))
        else:
            self.reduction = lambda x: x

    def forward(self, input, target=None):

        return self.reduction(self._p(self._mask*(input - self._target)))


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
torch.backends.cudnn.deterministic = True


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

        for n_wells, threshold in zip([4, 8, 20], [18, 12, 7]):
            print('... ... Number of wells:', n_wells)

            data = pd.read_csv(data_dir_path/'fluvgan_2_inversion_well-data.csv')

            is_valid = (data['Sample'] == i_sample) & (data['Well'] <= n_wells)
            X = torch.tensor(data.loc[is_valid, ['K', 'J', 'I']].to_numpy()).int().T
            y = torch.tensor(data.loc[is_valid, 'Fraction'].to_numpy()).float()


################################################################################
# Inference

            n_samples = 300

            z = torch.empty((n_samples, generator.nz))
            history = dict()
            for i in tqdm(range(n_samples)):

                history['loss ' + str(i + 1)] = []

                _z = torch.randn(1, generator.nz, 1, 1, 1, requires_grad=True)

                optimizer = torch.optim.Adam([_z], lr=0.01)
                loss_fn_prior = nn.BCEWithLogitsLoss()
                label = torch.tensor(1.)
                loss_fn_context = ContextLoss(X, 2.*y - 1., threshold, (16, 128, 128))

                for step in tqdm(range(1500), leave=False):
                    optimizer.zero_grad()
                    samples = generator(_z)
                    proba = discriminator(samples)
                    loss = loss_fn_context(samples[:, 0]) + 10.*loss_fn_prior(proba['data'], label)
                    loss.backward()
                    optimizer.step()
                    history['loss ' + str(i + 1)].append(loss.item())

                z[i] = _z[0, :, 0, 0, 0]

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

                    samples = 0.5*generator(z[s].view(z[s].shape + (1, 1, 1))) + 0.5

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

            label = 'fluvgan_2_inversion_3_prior-art_1_' + str(i_sample) + '_' + str(n_wells) + '_' + i_model + '_'

            history.to_csv(output_dir_path/(label + 'history.csv'), index=False)
            np.save(output_dir_path/(label + 'z.npy'), z.cpu().detach().numpy())
            error.to_csv(output_dir_path/(label + 'error.csv'), index=False)
