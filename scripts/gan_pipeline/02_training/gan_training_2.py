################################################################################
# Imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import os
import shutil
from functools import partial
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim

# Import Custom Modules
from dataloader import FaciesDataset
from utils import parse_hybrid_args, validate_dataset, save_config

# VoxGAN Imports
from voxgan.data.datasets import *
from voxgan.models.base import CustomGAN
from voxgan.networks import resnet
from voxgan.networks.utils import initialize_weights_normal
from voxgan.models.metrics import MSSWD, LoS

# ---------------------------------------------------------
# Custom Wasserstein Loss Functions (VoxGAN Compatible)
# ---------------------------------------------------------
def wasserstein_discriminator_loss(predictions, targets):
    """
    Wasserstein critic loss: D(fake) - D(real)
    VoxGAN calls this twice per batch (once for real images, once for fake images).
    Targets are 1.0 for real, 0.0 for fake.
    
    Math trick: (1 - 2 * targets) 
    -> If real (targets=1.0): returns -1 * predictions (minimizes real)
    -> If fake (targets=0.0): returns +1 * predictions (maximizes fake)
    """
    # Calculate the base Wasserstein directional loss
    w_loss = torch.mean(predictions * (1.0 - 2.0 * targets))
    
    # Drift penalty is only applied to real images (where targets=1.0)
    drift_penalty = 0.0001 * torch.mean((predictions ** 2) * targets)
    
    return w_loss + drift_penalty

def wasserstein_generator_loss(predictions, targets):
    """
    Wasserstein generator loss: -D(fake)
    VoxGAN passes (predictions, targets) automatically, but WGAN ignores the target labels.
    """
    return -torch.mean(predictions)

# ---------------------------------------------------------
# OOP Trainer Class
# ---------------------------------------------------------
class FluvialGANTrainer:
    """
    Object-Oriented wrapper for training a 3D Wasserstein GAN with Spectral Normalization.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Hardcoded Architecture Variables
        self.model_name = 'architecture_4_wgan_sn'
        self.nz = 100
        self.nl = (3, 5, 5)
        self.use_one_hot = True
        self.nc = 3  # Assuming 3 facies types based on prior scripts
        self.encoding_tag = "one_hot"
        self.num_training_runs = 1
        
        self._setup_environment()

    def _setup_environment(self):
        """Sets up seeds, directories, and validates inputs."""
        np.random.seed(43)
        torch.manual_seed(43)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

        self.run_dir = os.path.join(self.config['output_dir'], self.config['run_name'])
        os.makedirs(self.run_dir, exist_ok=True)
        
        config_save_path = os.path.join(self.run_dir, "config.json")
        save_config(self.config, config_save_path)
        print(f"Saved run configuration to: {config_save_path}")

        validate_dataset(self.config['data_file'])
        self.dataset_name = os.path.splitext(os.path.basename(self.config['data_file']))[0]

    def build_models(self):
        """Defines the Generator, Discriminator, and their corresponding Optimizers."""
        
        # Output is a probability distribution (Softmax) instead of Tanh [-1, 1]
        last_activation = partial(nn.Softmax, dim=1)

        self.generator = partial(resnet.DeepGenerator3d,
                                 nz=self.nz, ngf=64, nc=self.nc, nl=self.nl,
                                 max_factor=16, residual_weight=1., mode='nearest', kernel_size=3,
                                 layer_normalization=nn.BatchNorm3d,
                                 last_layer_normalization=nn.BatchNorm3d,
                                 weight_normalization=nn.utils.parametrizations.spectral_norm,
                                 activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                 last_activation=last_activation,
                                 use_double_conv=False, use_double_resblocks=False,
                                 use_attention=False, skip_z=False, split_z=False)

        self.discriminator = partial(resnet.DeepDiscriminator3d,
                                     ndf=64, nc=self.nc, nl=self.nl,
                                     max_factor=16, residual_weight=1., kernel_size=3,
                                     layer_normalization=None,
                                     weight_normalization=nn.utils.parametrizations.spectral_norm,
                                     activation=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
                                     use_double_conv=True, use_double_resblocks=False,                      
                                     use_attention=False)

        # Adam learning rates adjusted for Wasserstein with Spectral Norm
        self.optimizer_generator = partial(optim.Adam, lr=5e-4, betas=(0., 0.9))
        self.optimizer_discriminator = partial(optim.Adam, lr=5e-4, betas=(0., 0.9))

    def configure_metrics(self):
        """Initializes validation metrics (MSSWD, LoS)."""
        metric_params = dict(n_levels=3,
                             n_descriptors=1024,
                             descriptor_size=(4, 7, 7),
                             n_repeat=12,
                             n_proj=256,
                             padding_mode='circular', 
                             combine_levels=True, 
                             batch_size=self.config['val_batch_size'],
                             n_gpu=self.config['num_gpus'])

        ms_swd_fa1 = MSSWD(**metric_params, channel=0)
        ms_swd_fa2 = MSSWD(**metric_params, channel=1)
        ms_swd_fa3 = MSSWD(**metric_params, channel=2)
        los = LoS(channel=0, batch_size=self.config['val_batch_size'], n_gpu=self.config['num_gpus'])
        
        self.metrics = [ms_swd_fa1, ms_swd_fa2, ms_swd_fa3, los]

    def _get_dataset(self):
        """Returns the configured partial dataset."""
        return partial(FaciesDataset,
                       h5_path=self.config['data_file'],
                       num_samples=self.config['num_samples'],
                       save_mapping_dir=self.run_dir,
                       use_one_hot=self.use_one_hot,
                       dataset_name=self.dataset_name,
                       num_epochs=self.config['epochs'])

    def train(self):
        """Executes the main training loop."""
        dataset = self._get_dataset()
        
        for i in range(self.num_training_runs):
            output_label = f'wgan_training_{self.model_name}_{self.encoding_tag}_{i + 1}'
            
            gan = CustomGAN(self.generator,
                            self.discriminator,
                            output_dir_path=self.run_dir,
                            output_label=output_label,
                            verbose=2,
                            num_gpus=self.config['num_gpus'],
                            num_nodes=1,
                            distributed=False,
                            backend='nccl',
                            use_amp_training=True)
            
            # WGAN setup relies on Spectral Norm; R1 Gradient penalty is removed.
            gan.configure(self.optimizer_generator,
                          self.optimizer_discriminator,
                          wasserstein_generator_loss,
                          wasserstein_discriminator_loss,
                          initialize_weights=initialize_weights_normal,
                          num_iter_discriminator=1, 
                          num_accumulated=1, 
                          fake_label_generator=1.,
                          real_label_discriminator=1.0,
                          fake_label_discriminator=0.0,
                          penalty_generator=None,
                          penalty_discriminator=None) 
            
            gan.train(dataset,
                      num_epochs=self.config['epochs'],
                      batch_size=self.config['batch_size'],
                      num_workers=0 * self.config['num_gpus'],
                      pin_memory=True,
                      drop_last=True,
                      checkpoint_step=1e12,
                      sampling_step=100,
                      sampling_size=3,
                      metrics=self.metrics,
                      metric_step=100,
                      validation_size=self.config['validation_size'],
                      validation_batch_size=self.config['val_batch_size'],
                      preload_validation=True,
                      resume_checkpoint_id=None)

            self._post_process(output_label, i)

    def _post_process(self, output_label, run_index):
        """Handles checkpoint cleanup and initiates visualization generation."""
        checkpoint_dir_path = os.path.join(self.run_dir, f'{output_label}_Training_Checkpoints')
        checkpoint_paths = glob(os.path.join(checkpoint_dir_path, f'{output_label}_training_checkpoint_*'))
        
        if checkpoint_paths:
            checkpoint_paths = sorted(checkpoint_paths, key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
            last_checkpoint = checkpoint_paths[-1]

            new_filename = f"{self.model_name}_{self.dataset_name}_{self.encoding_tag}_epochs_{self.config['epochs']}_bs_{self.config['batch_size']}_run_{run_index + 1}.pt"
            final_checkpoint_path = os.path.join(self.run_dir, new_filename)

            shutil.copy(last_checkpoint, final_checkpoint_path)
            shutil.rmtree(checkpoint_dir_path)
            
            print(f"Saved final renamed model checkpoint to: {final_checkpoint_path}\n")

            print("-" * 35)
            print("Running post-training visualization and generation...")
            try:
                from visualize_and_generate import plot_losses, generate_realizations
                csv_path = os.path.join(self.run_dir, f"{output_label}_history.csv")
                plot_losses(csv_path, self.run_dir)
                generate_realizations(self.nc, final_checkpoint_path, self.run_dir, num_realizations=10)
                print("Post-training visualization complete.")
            except ImportError as e:
                print(f"Could not import visualization modules: {e}")
            except Exception as e:
                print(f"Error during post-training visualization: {e}")

    def run(self):
        """Master execution method."""
        print("--- Starting GAN Training Pipeline ---")
        self.build_models()
        self.configure_metrics()
        self.train()
        print("--- Pipeline Execution Complete ---")


if __name__ == '__main__':
    # Parse CLI arguments
    config = parse_hybrid_args()
    
    # Initialize and run the object-oriented pipeline
    trainer = FluvialGANTrainer(config)
    trainer.run()