"""
Architecture DCGAN
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator
             + R1 regularization

"""


################################################################################
# Imports
import warnings
# Ignore specific harmless warnings to clean up the terminal
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from voxgan.data.datasets import *
from voxgan.data.fluvdeposet import FluvDepoSetDataset
from voxgan.models.base import CustomGAN
from voxgan.networks import resnet
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.loss import R1Regularization
from voxgan.models.metrics import MSSWD, LoS
from scripts.old_scripts.old_gan_scripts.GAN.dataloader_old import FaciesDataset


################################################################################
# Paths
if os.name == 'nt':  # Windows
    global_path = os.getcwd()
    training_data_dir_path = Path(global_path,'datasets/training')
    output_dir_path = Path(global_path, 'outputs')

else: #Linux
    home_path = Path.home()
    training_data_dir_path = str(home_path / 'data' / 'training')
    output_dir_path = str(home_path / 'data' / 'outputs')
    os.makedirs(output_dir_path, exist_ok=True)

################################################################################
# Setting

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

use_one_hot = True  # Toggle this to switch between modes
num_classes = 3   
num_gpus = 2
num_training = 1
num_epochs = 50
batch_size = 4


################################################################################
# Model

model = 'architecture_4_dcgan_one_hot'

nz = 100
nc = num_classes 
nl = (4, 6, 6)

generator = partial(resnet.DeepGenerator3d,
                    nz=nz,
                    ngf=64,
                    nc=nc,
                    nl=nl,
                    max_factor=16,
                    residual_weight=1.,
                    mode='nearest',
                    kernel_size=3,
                    layer_normalization=nn.BatchNorm3d,
                    last_layer_normalization=nn.BatchNorm3d,
                    weight_normalization=nn.utils.parametrizations.spectral_norm,
                    activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                    last_activation=partial(nn.Softmax, dim=1),
                    use_double_conv=False,
                    use_double_resblocks=False,
                    use_attention=False,
                    skip_z=False,
                    split_z=False)
discriminator = partial(resnet.DeepDiscriminator3d,
                        ndf=64,
                        nc=nc,
                        nl=nl,
                        max_factor=16,
                        residual_weight=1.,
                        kernel_size=3,
                        layer_normalization=None,
                        weight_normalization=nn.utils.parametrizations.spectral_norm,
                        activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                        use_double_conv=False,
                        use_double_resblocks=False,
                        use_attention=False)
optimizer_generator = partial(optim.Adam,
                              lr=2e-4,
                              betas=(0., 0.99))
optimizer_discriminator = partial(optim.Adam,
                                  lr=2e-4,
                                  betas=(0., 0.99))
loss_generator = nn.BCEWithLogitsLoss()
loss_discriminator = nn.BCEWithLogitsLoss()
weight_initializer = initialize_weights_normal
num_iter_discriminator = 1
num_accumulated = 2
fake_label_generator = 1.
real_label_discriminator = 1.
fake_label_discriminator = 0.
penalty_generator = None
penalty_discriminator = R1Regularization(gamma=10., num_iter=16, use_amp=True)


################################################################################
# Validation

# Old validation over 1 channel
# ms_swd = MSSWD(n_levels=3,
#                n_descriptors=512,
#                descriptor_size=(3, 7, 7),
#                n_repeat=12,
#                n_proj=128,
#                padding_mode='zeros',
#                combine_levels=True,
#                batch_size=2,
#                n_gpu=num_gpus)
# los = LoS(channel=0, batch_size=batch_size, n_gpu=num_gpus)
# metrics = [ms_swd, los]

################################################################################
# Validation

metric_params = dict(n_levels=3,
                     n_descriptors=512,
                     descriptor_size=(3, 7, 7),
                     n_repeat=12,
                     n_proj=128,
                     padding_mode='constant',
                     combine_levels=True,
                     batch_size=4,         
                     n_gpu=num_gpus)

# Instantiate a separate metric for each Facies Association
ms_swd_fa1 = MSSWD(**metric_params, channel=0)
ms_swd_fa2 = MSSWD(**metric_params, channel=1)
ms_swd_fa3 = MSSWD(**metric_params, channel=2)

los = LoS(channel=0, batch_size=2, n_gpu=num_gpus)

metrics = [ms_swd_fa1, ms_swd_fa2, ms_swd_fa3, los]


################################################################################
# Dataset

transform = None#Compose([Crop(((1, 3), (8, 28), None, None)),
#                      FillNaN((0., 'max+1')),
#                      RandomCrop((None, 16, 128, 128)),
#                      Scale(((0, 1), None)),
#                      ToTensor()])

dataset = partial(FaciesDataset,
                  root=training_data_dir_path)


################################################################################
# Training

if __name__ == '__main__':
    for i in range(num_training):
        gan = CustomGAN(generator,
                        discriminator,
                        output_dir_path=output_dir_path,
                        output_label='fluvgan_1_training_1_' + model + '_' + str(i + 1),
                        verbose=2,
                        num_gpus=num_gpus,
                        num_nodes=1,
                        distributed=False,
                        backend='nccl',
                        use_amp_training=True)
        
        gan.configure(optimizer_generator,
                      optimizer_discriminator,
                      loss_generator,
                      loss_discriminator,
                      initialize_weights=weight_initializer,
                      num_iter_discriminator=num_iter_discriminator,
                      num_accumulated=num_accumulated,
                      fake_label_generator=fake_label_generator,
                      real_label_discriminator=real_label_discriminator,
                      fake_label_discriminator=fake_label_discriminator,
                      penalty_generator=penalty_generator,
                      penalty_discriminator=penalty_discriminator)
        
        gan.train(dataset,
                  num_epochs=num_iter_discriminator*num_epochs,
                  batch_size=batch_size,
                  num_workers=4*num_gpus,
                  pin_memory=True,
                  drop_last=True,
                  checkpoint_step=1e12,
                  sampling_step=100,
                  sampling_size=3,
                  metrics=metrics,
                  metric_step=100,
                  validation_size=0.1,
                  validation_batch_size=2,
                  preload_validation=True,
                  resume_checkpoint_id=None)


################################################################################
# Cleaning

        checkpoint_dir_path = output_dir_path + '/fluvgan_1_training_1_' + model + '_' + str(i + 1) + '_Training_Checkpoints'
        checkpoint_paths = glob(checkpoint_dir_path + '/fluvgan_1_training_1_' + model + '_' + str(i + 1) + '_training_checkpoint_*')
        checkpoint_paths = sorted(checkpoint_paths, key=lambda s: int(re.findall(r'\d+', str(s))[-1]))

        shutil.copy(checkpoint_paths[-1], output_dir_path)
        shutil.rmtree(checkpoint_dir_path)
