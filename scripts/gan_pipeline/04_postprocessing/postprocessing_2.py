import os
import pathlib
import glob
import json
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.stats import entropy
from scipy.ndimage import label
from scipy.spatial.distance import jensenshannon
from skimage.util import view_as_windows
from collections import Counter

import sys
from pathlib import Path

# Adjust path based on your environment
scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# Import custom utilities (Commented out if not available in standalone run)
# from gan_pipeline.core.custom_plots import apply_custom_plotting_flavor

class PostProcessing:
    """Handles spatial and statistical validation metrics for GAN-generated geological facies.

    This class loads 3D arrays of geological facies (both real and GAN-generated),
    computes spatial statistics (MPS, connectivity, normalized entropy, JSD), 
    and generates visual diagnostic plots including 3D renders.

    Attributes:
        output_dir (str): Directory where generated plots and CSVs will be saved.
        real_files (list): Sorted list of file paths for the real dataset.
        gan_files (list): Sorted list of file paths for the GAN-generated dataset.
        real_data (list): Loaded real data arrays in memory.
        gan_data (list): Loaded GAN data arrays in memory.
        facies_mapping (dict): Mapping from categorical indices to specific geological codes.
    """
    
    def __init__(self, output_dir, real_path, gan_path):
        """Initializes the class and locates files, but defers loading to save memory.

        Args:
            output_dir (str): Directory to save outputs (e.g., 'outputs/metrics').
            real_path (str): Glob pattern matching real data files (e.g., 'datasets/*.npy').
            gan_path (str): Glob pattern matching GAN data files (e.g., 'outputs/*.npy').
        """ 
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        self.real_files = sorted(glob.glob(str(real_path)))
        self.gan_files = sorted(glob.glob(str(gan_path)))
        
        self.real_data = []
        self.gan_data = []
        
        self.facies_mapping = {0: 1, 1: 4, 2: 8}
    
    def load_real_samples(self, limit=100):
        """Loads real data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading Real samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.real_files[:limit] if limit else self.real_files
        self.real_data = [self._load_real(f) for f in files_to_load]
        print(f"Loaded {len(self.real_data)} real samples into memory.")

    def load_gan_samples(self, limit=100):
        """Loads GAN data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading GAN samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.gan_files[:limit] if limit else self.gan_files
        self.gan_data = [self._load_gan(f) for f in files_to_load]
        print(f"Loaded {len(self.gan_data)} GAN samples into memory.")

    def _load_real(self, file):
        """Loads a facies array and maps real labels to geological codes.

        Args:
            file (str): File path to a saved .npy numpy array.

        Returns:
            numpy.ndarray: The 3D array of mapped geological facies.
        """
        data = np.load(file)
        mapped_data = np.zeros_like(data)
        mapped_data[(data >= 0) & (data < 4)] = 1
        mapped_data[(data >= 4) & (data < 8)] = 4
        mapped_data[(data >= 8)] = 8
        return mapped_data
    
    def _load_gan(self, file):
        """Loads GAN output and standardizes it to a 3D categorical array.

        Args:
            file (str): File path to a saved .npz or .npy array.

        Returns:
            numpy.ndarray: The 3D array of mapped geological facies.
            
        Raises:
            ValueError: If the input array does not conform to 3D or 4D expectations.
        """
        if file.endswith('.npz'):
            with np.load(file) as data:
                raw_arr = data['facies'] if 'facies' in data else data[data.files[0]]
        else:
            raw_arr = np.load(file)

        if raw_arr.ndim == 4:
            class_indices = np.argmax(raw_arr, axis=0) 
        elif raw_arr.ndim == 3:
            if raw_arr.min() < 0 or raw_arr.dtype in [np.float32, np.float64]:
                class_indices = np.round(raw_arr + 1).astype(int)
                class_indices = np.clip(class_indices, 0, 2)
            else:
                class_indices = raw_arr.astype(int)
        else:
            raise ValueError(f"Unexpected array shape {raw_arr.shape} in file {file}")

        return self._map_indices_to_facies(class_indices)

    def _map_indices_to_facies(self, class_indices):
        """Maps 0-indexed categorical data to specific geological facies codes.

        Args:
            class_indices (numpy.ndarray): Array containing 0-indexed categorical values.

        Returns:
            numpy.ndarray: Mapped array containing physical facies codes.
        """
        if class_indices.max() < 3:
            mapping = np.array([1, 4, 8]) 
        else:
            mapping = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
        return mapping[class_indices]

    def _get_global_proportions(self, data_list, target_facies=[1, 4, 8]):
        """Calculates the global probability of each facies across the entire dataset.

        Args:
            data_list (list): List of 3D numpy arrays containing geological facies.
            target_facies (list, optional): List of facies codes to calculate proportions for.
                Defaults to [1, 4, 8].

        Returns:
            list: Float probabilities of each target facies, summing to 1.0.
        """
        if not data_list: return [0.0] * len(target_facies)
        counts = {f: 0 for f in target_facies}
        total_voxels = sum(arr.size for arr in data_list)
        
        for arr in data_list:
            for f_val in target_facies:
                counts[f_val] += np.sum(arr == f_val)
                
        return [counts[f] / total_voxels for f in target_facies]

    def _calculate_h_max(self, proportions):
        """Calculates the theoretical maximum Shannon entropy for a given global distribution.

        Args:
            proportions (list): List of global probabilities for each facies class.

        Returns:
            float: Theoretical maximum entropy in bits.
        """
        probs = np.array(proportions)
        probs = probs[probs > 0] # Filter out strict zeros to avoid log2(0) errors
        return -np.sum(probs * np.log2(probs))

    def _create_colormap_and_legend(self):
        """Creates a ListedColormap and legend patches based on facies configuration.

        Returns:
            tuple: Contains the custom ListedColormap and a list of matplotlib patches.
                   Returns (None, None) if the config file is not found.
        """
        try:
            with open('scripts/gan_pipeline/core/facies_config.json','r') as f:
                facies_properties = json.load(f)
        except Exception:
            return None, None
        
        sorted_facies = sorted(facies_properties.items(), key=lambda item: item[1]['val'])
        color_list = [item[1]['color'] for item in sorted_facies]
        custom_cmap = ListedColormap(color_list)
        legend_patches = [
            mpatches.Patch(color=info['color'], label=name.replace('_', ' ').title()) 
            for name, info in sorted_facies
        ]
        return custom_cmap, legend_patches

    def get_pattern_counts(self, data_array, template_size=(3, 3, 3)):
        """Extracts and counts Multiple Point Statistics (MPS) patterns via sliding window.

        Args:
            data_array (numpy.ndarray): 3D volume to analyze.
            template_size (tuple, optional): Window size for pattern extraction.
                Defaults to (3, 3, 3).

        Returns:
            tuple: Arrays of unique patterns and their corresponding counts.
        """
        windows = view_as_windows(data_array, template_size)
        n_patches = np.prod(windows.shape[:3])
        patch_size = np.prod(template_size)
        flat_windows = windows.reshape(n_patches, patch_size)
        return np.unique(flat_windows, axis=0, return_counts=True)

    def analyze_connectivity(self, data_array, target_facies):
        """Calculates 3D structural connectivity sizes for a specific facies.

        Args:
            data_array (numpy.ndarray): 3D geological volume.
            target_facies (int): The integer code of the facies to analyze (e.g., 1 for channels).

        Returns:
            numpy.ndarray: Array containing the volume sizes of all connected structural blobs.
        """
        binary_mask = (data_array == target_facies).astype(int)
        structure = np.ones((3, 3, 3)) 
        labeled_array, _ = label(binary_mask, structure=structure)
        unique_ids, blob_sizes = np.unique(labeled_array, return_counts=True)
        return blob_sizes[1:] 

    def connectivity_and_pattern_analysis(self, target_val=1, sample_limit=10):
        """Executes MPS and Connectivity comparisons using loaded data and prints to console.

        Because 3D pattern extraction (view_as_windows) and connected component 
        labeling are computationally expensive, this limits the number of samples processed.

        Args:
            target_val (int, optional): Facies to target for connectivity analysis. Defaults to 1.
            sample_limit (int, optional): Maximum number of samples to process from the loaded data. 
                Defaults to 10. Pass None to process all currently loaded samples.
        """
        # 1. Fallback if data isn't loaded yet
        load_limit = sample_limit if sample_limit is not None else 10
        if not self.real_data or not self.gan_data:
            print(f"Warning: Data not loaded. Auto-loading up to {load_limit} samples...")
            self.load_real_samples(limit=load_limit)
            self.load_gan_samples(limit=load_limit)

        real_pattern_counter, gan_pattern_counter = Counter(), Counter()
        all_real_blob_sizes, all_gan_blob_sizes = [], []

        # 2. Slice the datasets based on the requested limit
        real_to_process = self.real_data[:sample_limit] if sample_limit else self.real_data
        gan_to_process = self.gan_data[:sample_limit] if sample_limit else self.gan_data

        # 3. Process Real Data
        print(f"Processing {len(real_to_process)} Real samples...")
        for data in real_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                real_pattern_counter[tuple(p)] += c 
            all_real_blob_sizes.extend(self.analyze_connectivity(data, target_val))

        # 4. Process GAN Data
        print(f"Processing {len(gan_to_process)} GAN samples...")
        for data in gan_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                gan_pattern_counter[tuple(p)] += c
            all_gan_blob_sizes.extend(self.analyze_connectivity(data, target_val))
        
        # 5. Output Results
        # 5. Output Results
        print("\n--- Results ---")
        print(f"Total unique patterns (Real): {len(real_pattern_counter)}")
        print(f"Total unique patterns (GAN): {len(gan_pattern_counter)}")
        
        if all_real_blob_sizes and all_gan_blob_sizes:
            real_max = max(all_real_blob_sizes)
            gan_max = max(all_gan_blob_sizes)
            real_med = np.median(all_real_blob_sizes)
            gan_med = np.median(all_gan_blob_sizes)
            real_mean = np.mean(all_real_blob_sizes)
            gan_mean = np.mean(all_gan_blob_sizes)

            print(f"\nMax blob size    (Real): {real_max} | (GAN): {gan_max}")
            print(f"Median blob size (Real): {real_med} | (GAN): {gan_med}")
            print(f"Mean blob size   (Real): {real_mean:.2f} | (GAN): {gan_mean:.2f}")
            
            # Return connectivity statistics instead of the repeated pattern metrics
            return {
                'real_max': real_max, 'gan_max': gan_max,
                'real_med': real_med, 'gan_med': gan_med,
                'real_mean': real_mean, 'gan_mean': gan_mean
            }
        else:
            print(f"\nNo blobs found for target facies value {target_val}!")
            return None

    def plot_facies_percentages(self, mode='gan', show_plot=True, save_plot=False):
        """Plots volume percentages of specific facies globally across the dataset.

        Args:
            mode (str, optional): Target to evaluate ('gan', 'real', or 'both'). Defaults to 'gan'.
            show_plot (bool, optional): If True, displays the plot interactively. Defaults to True.
            save_plot (bool, optional): If True, saves the plot to the output directory. Defaults to False.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        valid_modes = ['gan', 'real', 'both']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        if mode in ['gan', 'both'] and not self.gan_data:
            self.load_gan_samples()
        if mode in ['real', 'both'] and not self.real_data:
            self.load_real_samples()
            
        print(f"\n--- Generating Facies Distribution (Mode: {mode.upper()}) ---")

        target_facies = [1, 4, 8]
        labels = ["Channel", "Crevasse Splay/Levee", "Floodplain"]
        colors = ['#f1970f', '#fffc65', '#33ff00']
        
        def calculate_percentages(data_list):
            probs = self._get_global_proportions(data_list, target_facies)
            return [p * 100 for p in probs]

        fig, ax = plt.subplots(figsize=(10, 6))

        if mode == 'both':
            real_percentages = calculate_percentages(self.real_data)
            gan_percentages = calculate_percentages(self.gan_data)
            
            x = np.arange(len(labels))
            width = 0.35 
            
            bars_real = ax.bar(x - width/2, real_percentages, width, color=colors, edgecolor='black', alpha=0.5)
            bars_gan = ax.bar(x + width/2, gan_percentages, width, color=colors, edgecolor='black', hatch='//')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            
            for bar in bars_real:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, color='#555555')
            for bar in bars_gan:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
            legend_elements = [
                mpatches.Patch(facecolor='gray', alpha=0.5, edgecolor='black', label=f'Real Data ({len(self.real_data)} samples)'),
                mpatches.Patch(facecolor='gray', hatch='//', edgecolor='black', label=f'GAN Data ({len(self.gan_data)} samples)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
            ax.set_title('Facies Volume Distribution: Real vs. GAN', fontsize=14, pad=15)
            
        else:
            target_data = self.gan_data if mode == 'gan' else self.real_data
            percentages = calculate_percentages(target_data)
            
            bars = ax.bar(labels, percentages, color=colors, edgecolor='black', linewidth=1.2)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
            title_prefix = "GAN" if mode == 'gan' else "Real"
            ax.set_title(f'{title_prefix} Facies Volume Distribution ({len(target_data)} Samples)', fontsize=14, pad=15)

        ax.set_ylabel('Volume Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100) 
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plot:
            filename = f"facies_distribution_{mode}.png"
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved {mode} plot to: {plot_path}")
            
        if show_plot: plt.show()
        else: plt.close(fig)

    def plot_entropy(self, data_source='gan', axis=None, num_slices=9, plot_title=True, show_plot=True, save_plot=False):
        """Calculates and plots cell-wise normalized spatial entropy across the dataset.

        Args:
            data_source (str, optional): Target to evaluate ('gan' or 'real'). Defaults to 'gan'.
            axis (str, optional): Axis to slice ('X', 'Y', 'Z'). If None, processes all. Defaults to None.
            num_slices (int, optional): Slices to plot per axis (1, 3, or 9). Defaults to 9.
            show_plot (bool, optional): If True, displays the plot interactively. Defaults to True.
            save_plot (bool, optional): If True, saves the plot to the output directory. Defaults to False.

        Raises:
            ValueError: If input arguments do not match expected constraints.
        """
        valid_sources = ['gan', 'real']
        if data_source not in valid_sources:
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
            
        valid_slices = [1, 3, 9]
        if num_slices not in valid_slices:
            raise ValueError(f"num_slices must be 1, 3, or 9. Received: {num_slices}")
            
        if axis:
            axis = axis.upper()
            if axis not in ['X', 'Y', 'Z']:
                raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
        
        if data_source == 'gan' and not self.gan_data:
            self.load_gan_samples()
        elif data_source == 'real' and not self.real_data:
            self.load_real_samples()
            
        target_data = self.gan_data if data_source == 'gan' else self.real_data
        num_total = len(target_data)
        axes_to_plot = [axis] if axis else ['Z', 'Y', 'X']

        # Determine theoretical max entropy for THIS specific dataset
        global_probs = self._get_global_proportions(target_data)
        h_max = self._calculate_h_max(global_probs)
        
        print(f"\n--- Generating {data_source.upper()} Normalized Entropy Matrices ---")
        print(f"Dataset Global Proportions: {[round(p, 4) for p in global_probs]}")
        print(f"Dataset Theoretical H_max:  {h_max:.4f} bits")

        nz, ny, nx = target_data[0].shape
        dims = {'Z': nz, 'Y': ny, 'X': nx}
        slices_dict = {ax: sorted(random.sample(range(dims[ax]), num_slices)) for ax in axes_to_plot}

        stacks = {}
        if 'Z' in axes_to_plot: stacks['Z'] = np.zeros((num_total, num_slices, ny, nx), dtype=np.uint8) 
        if 'Y' in axes_to_plot: stacks['Y'] = np.zeros((num_total, num_slices, nz, nx), dtype=np.uint8) 
        if 'X' in axes_to_plot: stacks['X'] = np.zeros((num_total, num_slices, nz, ny), dtype=np.uint8) 

        for i, data_3d in enumerate(target_data):
            if 'Z' in stacks: stacks['Z'][i] = data_3d[slices_dict['Z'], :, :]
            if 'Y' in stacks: stacks['Y'][i] = data_3d[:, slices_dict['Y'], :].swapaxes(0, 1)
            if 'X' in stacks: stacks['X'][i] = data_3d[:, :, slices_dict['X']].transpose(2, 0, 1)

        # Pass h_max to the helper
        if 'Z' in stacks: self._plot_entropy_helper(stacks['Z'], slices_dict['Z'], 'Z', 'X', 'Y', data_source, h_max, plot_title, show_plot, save_plot)
        if 'Y' in stacks: self._plot_entropy_helper(stacks['Y'], slices_dict['Y'], 'Y', 'X', 'Z', data_source, h_max, plot_title, show_plot, save_plot)
        if 'X' in stacks: self._plot_entropy_helper(stacks['X'], slices_dict['X'], 'X', 'Y', 'Z', data_source, h_max, plot_title, show_plot, save_plot)

    def _plot_entropy_helper(self, slices_stack, slice_indices, axis_name, xlabel, ylabel, data_source, h_max, plot_title, show_plot, save_plot):
        """Internal helper to compute, format, and render normalized entropy charts.

        Args:
            slices_stack (numpy.ndarray): Stacked 2D cross-sections across all realizations.
            slice_indices (list): Integer indices indicating absolute slice depth in the 3D grid.
            axis_name (str): Label of the slicing axis ('X', 'Y', or 'Z').
            xlabel (str): Label for the horizontal axis of the output plot.
            ylabel (str): Label for the vertical axis of the output plot.
            data_source (str): Identifier string ('gan' or 'real') for title formatting.
            h_max (float): The theoretical maximum entropy (bits) for normalization.
            show_plot (bool): Display plot interactively if True.
            save_plot (bool): Save plot to output directory if True.
        """
        num_realizations, n_slices, dim_y, dim_x = slices_stack.shape
        facies_values = [1, 4, 8]
        norm = mcolors.Normalize(vmin=0, vmax=1.0) 

        if n_slices == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes_list = [axes]  
        elif n_slices == 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes_list = axes.flatten()
        else: 
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes_list = axes.flatten()

        for idx, slice_val in enumerate(slice_indices):
            probs = np.zeros((len(facies_values), dim_y, dim_x))
            for i, f_val in enumerate(facies_values):
                probs[i] = np.sum(slices_stack[:, idx, :, :] == f_val, axis=0) / num_realizations
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entropy_map = entropy(probs, base=2, axis=0)
                # Normalize the map using H_max
                if h_max > 0:
                    entropy_map = entropy_map / h_max
                
            im = axes_list[idx].imshow(entropy_map, cmap='magma', origin='lower', norm=norm)
            axes_list[idx].set_title(f"{axis_name}-Slice = {slice_val}")
            
            if n_slices == 1 or (n_slices == 3 and idx == 0) or (n_slices == 9 and idx % 3 == 0):
                axes_list[idx].set_ylabel(ylabel)
            if n_slices == 1 or n_slices == 3 or (n_slices == 9 and idx >= 6):
                axes_list[idx].set_xlabel(xlabel)

        if n_slices == 1:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        elif n_slices == 3:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        else:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            
        fig.colorbar(im, cax=cbar_ax).set_label('Normalized Entropy (0 to 1)', rotation=270, labelpad=15)
        
        plane = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}[axis_name]
        if plot_title:
            plt.suptitle(f"{data_source.upper()} Data: {plane} Plane Normalized Cell-Wise Entropy ({num_realizations} Realizations)", fontsize=16)

        if save_plot:
            path = os.path.join(self.output_dir, f"norm_entropy_{data_source}_{plane}_{num_realizations}_samples_{n_slices}_slices.png")
            plt.savefig(path, bbox_inches='tight', dpi=600)
            print(f"Saved normalized entropy plot to: {path}")
            
        if show_plot: plt.show()
        else: plt.close(fig)
    
    def compute_slice_metrics(self, axis='Z', num_slices=3):
        """Computes slice-wise aggregate scalar metrics comparing GAN to Real distributions.
        
        Evaluates Mean Normalized Entropy (to test diversity) and Jensen-Shannon 
        Divergence (JSD) (to test fidelity/spatial accuracy) per slice. Both metrics 
        dynamically account for the theoretical H_max of their respective underlying datasets.

        Args:
            axis (str, optional): Target axis to slice ('X', 'Y', 'Z'). Defaults to 'Z'.
            num_slices (int, optional): Evenly distributed slices to extract. Defaults to 3.

        Returns:
            pandas.DataFrame: DataFrame containing Slice Index, Mean Normalized Entropy 
                (Real & GAN), and Real vs. GAN JSD scores. 
                Also saves directly to output_dir as a CSV.
        """
        if not self.gan_data: self.load_gan_samples()
        if not self.real_data: self.load_real_samples()
            
        print(f"\n--- Computing Scalar Metrics for {axis}-Axis Slices ---")
        
        # Calculate respective theoretical maximums for scaling
        real_global_probs = self._get_global_proportions(self.real_data)
        real_h_max = self._calculate_h_max(real_global_probs)
        
        gan_global_probs = self._get_global_proportions(self.gan_data)
        gan_h_max = self._calculate_h_max(gan_global_probs)
        
        facies_values = [1, 4, 8]
        num_real_samples = len(self.real_data)
        num_gan_samples = len(self.gan_data)
        
        nz, ny, nx = self.real_data[0].shape
        dims = {'Z': nz, 'Y': ny, 'X': nx}
        slice_indices = np.linspace(0, dims[axis] - 1, num_slices, dtype=int)
        
        results = []

        for slice_idx in slice_indices:
            if axis == 'Z':
                real_slices = np.array([d[slice_idx, :, :] for d in self.real_data])
                gan_slices = np.array([d[slice_idx, :, :] for d in self.gan_data])
            elif axis == 'Y':
                real_slices = np.array([d[:, slice_idx, :] for d in self.real_data])
                gan_slices = np.array([d[:, slice_idx, :] for d in self.gan_data])
            else: 
                real_slices = np.array([d[:, :, slice_idx] for d in self.real_data])
                gan_slices = np.array([d[:, :, slice_idx] for d in self.gan_data])
                
            dim_y, dim_x = real_slices.shape[1], real_slices.shape[2]
            
            p_real = np.zeros((len(facies_values), dim_y, dim_x))
            p_gan = np.zeros((len(facies_values), dim_y, dim_x))
            
            for i, f_val in enumerate(facies_values):
                p_real[i] = np.sum(real_slices == f_val, axis=0) / num_real_samples
                p_gan[i] = np.sum(gan_slices == f_val, axis=0) / num_gan_samples
                
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entropy_real = np.nanmean(entropy(p_real, base=2, axis=0))
                if real_h_max > 0: entropy_real /= real_h_max
                
                entropy_gan = np.nanmean(entropy(p_gan, base=2, axis=0))
                if gan_h_max > 0: entropy_gan /= gan_h_max
                
            p_real_flat = p_real.reshape(len(facies_values), -1)
            p_gan_flat = p_gan.reshape(len(facies_values), -1)
            
            js_distance = jensenshannon(p_real_flat.T, p_gan_flat.T, base=2)
            mean_jsd = np.nanmean(js_distance ** 2)
            
            results.append({
                'Axis': axis,
                'Slice_Index': slice_idx,
                'Real_Norm_Entropy': entropy_real,
                'GAN_Norm_Entropy': entropy_gan,
                'JSD_Real_vs_GAN': mean_jsd
            })
            
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        csv_path = os.path.join(self.output_dir, f"slice_metrics_normalized_{axis}.csv")
        df_results.to_csv(csv_path, index=False)
        
        return df_results

    def plot_3d_pyvista(self, data_source='gan', mode='separate', show_legend=True, show_plot=True, save_plot=False):
        """Renders an interactive or static 3D volumetric plot of a random geological sample.

        Args:
            data_source (str, optional): Target dataset to sample ('gan' or 'real'). Defaults to 'gan'.
            mode (str, optional): Visual layout mode ('separate' for 1x3 subplots, or 'combined' for 
                a single overlaid volume). Defaults to 'separate'.
            show_plot (bool, optional): If True, opens an interactive PyVista window. Defaults to True.
            save_plot (bool, optional): If True, saves an off-screen screenshot. Defaults to False.

        Raises:
            ValueError: If input arguments do not match expected constraints.
        """
        valid_sources = ['gan', 'real']
        valid_modes = ['separate', 'combined']
        
        if data_source not in valid_sources:
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")
            
        if data_source == 'gan' and not self.gan_data:
            self.load_gan_samples()
        elif data_source == 'real' and not self.real_data:
            self.load_real_samples()
            
        print(f"\n--- Generating 3D PyVista Plot ({data_source.upper()} Data | Mode: {mode.upper()}) ---")
        try:
            import pyvista as pv
            pv.set_jupyter_backend('static') # or use 'trame' if you have it installed for interactive 3D rotation

        except ImportError:
            print("Error: 'pyvista' is not installed. Skipping 3D plot.")
            return

        target_data = self.gan_data if data_source == 'gan' else self.real_data
        random_idx = random.randint(0, len(target_data) - 1)
        data_3d = target_data[random_idx]
        nz, ny, nx = data_3d.shape

        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.cell_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

        facies_colors = {1: '#f1970f', 4: '#fffc65', 8: '#33ff00'}
        facies_titles = {1: "Channel", 4: "Crevasse Splay/Levee", 8: "Floodplain"}
        
        if mode == 'separate':
            plotter = pv.Plotter(shape=(1, 3), image_scale=4, off_screen=save_plot and not show_plot, window_size=(7200, 2400))
        else:
            plotter = pv.Plotter(shape=(1, 1), image_scale=4, off_screen=save_plot and not show_plot, window_size=(2400, 2400))
            
        for i, f_val in enumerate([1, 4, 8]):
            if mode == 'separate':
                plotter.subplot(0, i)
                plotter.add_text(facies_titles[f_val], font_size=200, color='black', shadow=True)
                
            threshed = grid.threshold([f_val - 0.5, f_val + 0.5], scalars='Facies')
            
            if threshed.n_points > 0:
                plotter.add_mesh(
                    threshed, 
                    color=facies_colors[f_val], 
                    show_edges=False, 
                    ambient=0.2,
                    diffuse=0.8,
                    label=facies_titles[f_val] if mode == 'combined' else None
                )
                
            if mode == 'separate':
                plotter.view_isometric()

        if (mode == 'combined') and (show_legend == True):
            plotter.add_legend(bcolor='grey', face=None, size=(0.2, 0.2))
            plotter.view_isometric()

        plot_path = os.path.join(self.output_dir, f"3d_plot_{data_source}_sample_{random_idx}_{mode}.png")
        
        if save_plot:
            plotter.show(screenshot=plot_path)
            print(f"Saved 3D plot to: {plot_path}") 
        elif show_plot:
            plotter.show()
        else:
            plotter.close()

if __name__ == "__main__":
    # 1. Initialize
    validator = PostProcessing(
        output_dir='outputs/20000_training_samples',
        real_path='datasets/training/*.npy',
        gan_path='outputs/realizations/*.npy'
    )

    # 2. Load the Samples into self (Limiting to 10 to save RAM!)
    validator.load_gan_samples(limit=10)
    validator.load_real_samples(limit=10)

    # 3. Call functions with explicit save/show logic
    validator.plot_facies_percentages(show_plot=True, save_plot=False)
    validator.connectivity_and_pattern_analysis(target_val=1)
    validator.plot_entropy(show_plot=False, save_plot=True)
    validator.compute_slice_metrics(axis='Z', num_slices=3)
    validator.plot_3d_pyvista(show_plot=True, save_plot=True)