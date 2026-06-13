#!/usr/bin/env python3
import re
import h5py
import json
import argparse
from pathlib import Path
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

import torch
import torch.nn as nn

from voxgan.networks import resnet
from voxgan.data.datasets import Compose, Crop, FillNaN, Scale, ToTensor
from voxgan.models.metrics import MSSWD

def main():
    # ==========================================
    # 1. PATHS SETUP (Update these for your cluster)
    # ==========================================
    model_dir_path = Path('./outputs/post_training/20000_training_samples/RUN_10000_of_20000_samples_128xy_dataset_50_epochs_bs_32_val_size_020_onr_hot')
    test_data_dir_path = Path('./datasets/testing/testing_dataset_upper_plain_delta_128')
    output_dir_path = Path('./plots/post_training_plots/RUN_10000_of_20000_samples_128xy_dataset_50_epochs_bs_32_val_size_020_onr_hot')

    # Ensure the output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 2. SETTINGS & GPU SETUP
    # ==========================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    np.random.seed(42)
    torch.manual_seed(42)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpu_for_metric = min(2, available_gpus) if available_gpus > 0 else 0
    print(f"Using device: {device}")
    print(f"Available GPUs: {available_gpus}, allocated for MSSWD: {n_gpu_for_metric}")

    # ==========================================
    # 3. CHECKPOINT DETECTION
    # ==========================================
    checkpoint_paths = []
    for i in range(3):
        model_name = 'fluvgan_1_training_1_architecture_dcgan_4_' + str(i + 1) + '_training_checkpoint_*'
        paths = sorted(list(model_dir_path.glob(model_name)), key=lambda s: int(re.findall(r'\d+', str(s))[-1]))
        if paths:
            checkpoint_paths.append(paths[-1])

    if not checkpoint_paths:
        print("Default checkpoint naming pattern not matched. Searching for any .pt files...")
        checkpoint_paths = sorted(list(model_dir_path.glob("*.pt")))

    n_models = len(checkpoint_paths)
    if n_models == 0:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in: {model_dir_path}")

    print(f"Found {n_models} model(s) to evaluate:")
    for idx, path in enumerate(checkpoint_paths):
        print(f"  Model {idx + 1}: {path.name}")

    # ==========================================
    # 4. DATASET DEFINITION & LOADING
    # ==========================================
    class NpyOrH5Dataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None, mapping_config_path=None):
            self.root = Path(root)
            self.transform = transform
            self.file_paths = sorted(list(self.root.glob("*.npy")) + list(self.root.glob("*.h5")))
            self.len = len(self.file_paths)
            
            self.mapping = None
            if mapping_config_path and Path(mapping_config_path).exists():
                try:
                    with open(mapping_config_path, 'r') as f:
                        config = json.load(f)
                    self.mapping = np.array(config['forward_mapping_array'])
                    print(f"Loaded facies mapping array of length {len(self.mapping)}.")
                except Exception as e:
                    print(f"Failed to load mapping config: {e}")
            
            if self.mapping is None:
                self.mapping = np.zeros(15, dtype=np.int64)
                self.mapping[1:4] = 0
                self.mapping[4:8] = 1
                self.mapping[8:15] = 2
                
        def __len__(self):
            return self.len
            
        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            if file_path.suffix == '.npy':
                data = np.load(file_path)
            elif file_path.suffix == '.h5':
                with h5py.File(file_path, 'r') as file:
                    data = file['model'][:]
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            if len(data.shape) == 3:
                mapped_data = self.mapping[data.astype(np.int64)]
                num_classes = int(self.mapping.max() + 1)
                one_hot = np.eye(num_classes)[mapped_data]      
                one_hot = np.moveaxis(one_hot, -1, 0)           
                data = (one_hot * 2.0) - 1.0
                
            sample = {'data': data}
            if self.transform is not None:
                sample = self.transform(sample)
            return sample

    # Note: If your input data shape caused the previous mismatch runtime error, 
    # adjust the Crop parameters below to fit your cluster dataset files cleanly.
    transform = [
        Compose([Crop(((1, 3), (8, 24), (0, 128), None)), FillNaN((0., 'max+1')), Scale(((0, 1), None)), ToTensor()]),
        Compose([Crop(((1, 3), (12, 28), (72, 200), None)), FillNaN((0., 'max+1')), Scale(((0, 1), None)), ToTensor()]),
    ]

    mapping_config_path = model_dir_path / 'facies_mapping_config.json'
    
    # Pre-calculating dimensions using first slice setup
    base_dataset = NpyOrH5Dataset(test_data_dir_path, transform=transform[0], mapping_config_path=mapping_config_path)
    n_samples = len(transform) * len(base_dataset)
    
    # High-level tracking containers
    embedding = np.full((n_models * 2 * n_samples, 100 + 5), np.nan)
    test_samples = torch.empty((n_samples, 2, 16, 128, 128))

    print("Loading test dataset slices...")
    for i in range(len(transform)):
        dataset = NpyOrH5Dataset(test_data_dir_path, transform=transform[i], mapping_config_path=mapping_config_path)
        for j, test_sample in enumerate(tqdm(dataset, desc=f"Transform {i+1}", leave=False)):
            test_samples[i * len(dataset) + j] = test_sample['data']
            
            filename = Path(dataset.file_paths[j]).stem
            match = re.search(r'\d+', filename)
            sample_id = int(match.group()) if match else j
            
            for k in range(n_models):
                idx_pos = k * 2 * n_samples + i * len(dataset) + j
                embedding[idx_pos, 0] = k + 1
                embedding[idx_pos, 1] = sample_id
                embedding[idx_pos, 2] = i + 1

    # ==========================================
    # 5. GENERATION & METRIC PROCESSING
    # ==========================================
    batch_size = 100  # On a cluster with more VRAM, you can easily scale this up to 200+
    distances = np.zeros((n_models, 2 * n_samples, 2 * n_samples))

    for i in range(n_models):
        checkpoint_path = checkpoint_paths[i]
        print(f"\n--- Processing Model {i + 1}/{n_models} ({checkpoint_path.name}) ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        generator = resnet.DeepGenerator3d(nz=100,
                                           ngf=64,
                                           nc=3,
                                           nl=(3, 5, 5),
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
        generator.requires_grad_(False)

        with torch.no_grad():
            dummy_z = torch.zeros((1, generator.nz))
            dummy_out = generator(dummy_z.view(dummy_z.shape + (1, 1, 1)))
            generator_nc = dummy_out.shape[1]
        print(f"Generator output channels: {generator_nc}")

        generated_samples = torch.empty((n_samples, 2, 16, 128, 128))
        with torch.no_grad():
            for j in tqdm(range(0, n_samples, batch_size), desc="Generating realizations"):
                current_batch_size = min(batch_size, n_samples - j)
                z = torch.randn((current_batch_size, generator.nz))
                s = slice(i * 2 * n_samples + n_samples + j,
                          i * 2 * n_samples + n_samples + min(j + batch_size, n_samples))
                embedding[s, 0] = i + 1
                embedding[s, 3:generator.nz + 3] = z.cpu().numpy()
                
                gen_out = generator(z.view(z.shape + (1, 1, 1)))
                
                if generator_nc == 3:
                    gen_out = gen_out[:, 1:3]
                    
                generated_samples[j:min(j + batch_size, n_samples)] = gen_out

        # Combine real reference data with generated realizations
        combined_samples = torch.cat((test_samples, generated_samples))

        # Setup metric evaluator
        ms_swd = MSSWD(n_levels=3,
                       n_descriptors=512,
                       descriptor_size=(3, 7, 7),
                       n_repeat=12,
                       n_proj=128,
                       padding_mode='circular',
                       combine_levels=True,
                       n_gpu=n_gpu_for_metric)

        print("Computing pairwise MS-SWD distances...")
        for j in tqdm(range(combined_samples.shape[0]), desc="Outer Loop"):
            for k in range(j + 1, combined_samples.shape[0]):
                distance = ms_swd(combined_samples[j:j + 1], combined_samples[k:k + 1])
                distances[i, j, k] = distance
                distances[i, k, j] = distance

        print("Reducing dimensions using MDS...")
        reducer = MDS(n_components=2,
                      n_jobs=8,
                      random_state=42,
                      dissimilarity='precomputed')
        embedding[i * 2 * n_samples:(i + 1) * 2 * n_samples, -2:] = reducer.fit_transform(distances[i])

        print(f"Model {i + 1} MDS stress: {reducer.stress_}")
        
        # Clean VRAM caching between model calculations
        del generator
        torch.cuda.empty_cache()

    # ==========================================
    # 6. EXPORTING GENERATED METRICS
    # ==========================================
    print("\nProcessing results into dataframes...")
    columns = ['Model', 'Test_sample', 'Transform']
    columns += ['Latent_' + str(i + 1) for i in range(100)]
    columns += ['Dimension_1', 'Dimension_2']
    
    embedding_df = pd.DataFrame(data=embedding, columns=columns)
    embedding_df = embedding_df.convert_dtypes()

    csv_out = output_dir_path / 'fluvgan_1_training_2_test-msswd_embedding.csv'
    npy_out = output_dir_path / 'fluvgan_1_training_2_test-msswd_distance.npy'

    embedding_df.to_csv(csv_out, index=False)
    np.save(npy_out, distances)

    print(f"SUCCESS: Saved embedding table to: {csv_out}")
    print(f"SUCCESS: Saved distance matrix to: {npy_out}")

if __name__ == '__main__':
    main()