"""
The architecture of this model are based on Rongier & Peeters., 2025.

Architecture DCGAN
             + LeakyReLU in the generator
             + Binary cross entropy with logits as loss
             + Beta 1 of 0 & beta 2 of 0.99
             + Spectral normalization
             + Residual blocks
             + No batch normalization in the discriminator
             + R1 regularization

"""

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

# path definintion
home_path = Path.home()
training_data_dir_path = str(home_path / 'data' / 'training')
output_dir_path = str(home_path / 'workspace' / 'outputs')
os.makedirs(output_dir_path, exist_ok=True)

######## Settings ##########
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


num_gpus = 2
num_training = 1
num_epochs = 50
batch_size = 8

