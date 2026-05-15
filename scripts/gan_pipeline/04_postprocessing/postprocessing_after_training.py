"""
The purpose of this file is to validate the realization produced by the GAN model after the initial training phase. 
This is to ensure that the model is producing realistic samples and to visualize the training losses to understand the training dynamics.
The scripts includes the following functions:
1. Entropy Calculation over all generated realizations to assess the diversity of the generated samples.
2. MS-SWD calculation to evaluate the similarity between the generated samples and the real samples in terms of their distribution.
3. MPS calculation to check the nr of modes output by the GAN model
4. 3D visualization of generated samples, isolating FA1 to perform visual inspection of generated samples.
"""

import os
import pathlib
import glob
import json
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.stats import entropy
from scipy.ndimage import label
from skimage.util import view_as_windows
from collections import Counter

class PostProcessing1:
    """
    
    """
    def __init__(self, output_dir, real_path, gan_path, num_samples=10):
        """
        Initializes the PostProcessing1 class by loading file paths for real and GAN-generated samples.
        """
        self.output_dir = output_dir
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        self.path = os.path.relpath(self.script_path, self.cwd)
        self.real_files = glob.glob(real_path)[:num_samples]
        self.gan_files = glob.glob(gan_path)[:num_samples]
        self.facies_mapping = {0: 1, 1: 4, 2: 8}
        
        # Store num_samples dynamically based on what was actually found
        self.num_samples = min(num_samples, len(self.real_files), len(self.gan_files))

    def _load_real(self, file):
        """
        Loads the facies array and maps real labels to representative facies [1, 4, 8].
        """
        data = np.load(file)
        mapped_data = np.zeros_like(data)
        mapped_data[(data >= 0) & (data < 4)] = 1
        mapped_data[(data >= 4) & (data < 8)] = 4
        mapped_data[(data >= 8)] = 8
        
        return mapped_data

    def _load_gan(self, file):
        """
        Loads the facies array and maps GAN labels [-1, 0, 1] back to representative facies [1, 4, 8].
        """
        if file.endswith('.npz'):
            with np.load(file) as data:
                arr = data['facies'] if 'facies' in data else data[data.files[0]]
        else:
            arr = np.load(file)

        class_indices = np.round(arr + 1).astype(int)
        class_indices = np.clip(class_indices, 0, 2)
        mapping = np.array([1, 4, 8])

        return mapping[class_indices]

    def get_pattern_counts(self, data_array, template_size=(3, 3, 3)):
        """
        Extracts patterns and their raw counts from a single 3D array.
        """
        windows = view_as_windows(data_array, template_size)
        
        n_patches = np.prod(windows.shape[:3])
        patch_size = np.prod(template_size)
        flat_windows = windows.reshape(n_patches, patch_size)
        
        unique_patterns, counts = np.unique(flat_windows, axis=0, return_counts=True)
        return unique_patterns, counts

    def analyze_connectivity(self, data_array, target_facies):
        """
        Finds connected 3D bodies of a specific facies and returns their volumes.
        """
        binary_mask = (data_array == target_facies).astype(int)
        structure = np.ones((3, 3, 3)) 
        
        labeled_array, num_features = label(binary_mask, structure=structure)
        unique_ids, blob_sizes = np.unique(labeled_array, return_counts=True)
        
        # Remove the background (ID 0)
        return blob_sizes[1:] 
    
    def _plot_entropy_matrix_helper(self, slices_stack, slice_indices, axis_name, xlabel, ylabel, output_directory, data_dir, num_files):
        """
        Helper function to calculate and plot the entropy matrices for a given set of slices.
        
        Args:
            slices_stack (np.ndarray): 3D array of stacked slices from multiple realizations.
            slice_indices (list): List of slice indices to plot.
            axis_name (str): Name of the axis ('X', 'Y', or 'Z') being sliced.
            xlabel (str): Label for the X-axis of the plot.
            ylabel (str): Label for the Y-axis of the plot.
            output_directory (str): Directory to save the plot.
            data_dir (str): Original data directory (used for naming the output file).
            num_files (int): Number of sampled realizations.
        """
        path_to_facies_config = os.path.join(self.cwd,'scripts','gan_pipeline','core','facies_config.json')
        with open(path_to_facies_config, 'r') as json_file:
            facies_config = json.load(json_file)

        num_realizations, _, dim_y, dim_x = slices_stack.shape
        num_facies = len(facies_config)
        
        vmin, vmax = 0, 3.0 
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, slice_val in enumerate(slice_indices):
            probabilities = np.zeros((num_facies, dim_y, dim_x))
            for f_val in range(num_facies):
                probabilities[f_val] = np.sum(slices_stack[:, idx, :, :] == f_val, axis=0) / num_realizations

            entropy_map = entropy(probabilities, base=2, axis=0)
            im = axes[idx].imshow(entropy_map, cmap='magma', origin='lower', norm=norm)
            
            axes[idx].set_title(f"{axis_name}-Slice = {slice_val}", fontsize=12)
            axes[idx].set_xlabel(xlabel)
            axes[idx].set_ylabel(ylabel)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Entropy (bits)', rotation=270, labelpad=15)

        plane_names = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}
        plane = plane_names[axis_name]
        plt.suptitle(f"{plane} Plane Cell-Wise Entropy (Fixed Scale: {vmin}-{vmax} bits)\n({num_files} Realizations)", fontsize=16)

        dataset_name = os.path.basename(os.path.normpath(data_dir))
        plot_path = os.path.join(output_directory, f"entropy_matrix_{plane}_{dataset_name}_{num_files}_samples.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved {plane} entropy plot to: {plot_path}")

    def plot_entropy(self, data_dir, output_dir, all_files, num_files):
        """
        Samples a subset of files and plots the cell-wise entropy across multiple slices
        for X, Y, and Z planes to measure variability among realizations.
        
        Args:
            data_dir (str): Directory containing the .npz files.
            output_dir (str): Directory to save the resulting plots.
            all_files (list): List of all .npz file names.
            num_files (int): Number of realizations to sample.
        """
        print("\n--- Generating Entropy Matrices ---")
        num_to_sample = min(num_files, len(all_files))
        sampled_files = random.sample(all_files, num_to_sample)
        
        first_file_path = os.path.join(data_dir, sampled_files[0])
        sample_data = self._load_gan(first_file_path)
        max_z, ny, nx = sample_data.shape
        
        random_z_slices = sorted(random.sample(range(max_z), 9))
        random_y_slices = sorted(random.sample(range(ny), 9))
        random_x_slices = sorted(random.sample(range(nx), 9))

        stack_xy = np.zeros((num_to_sample, 9, ny, nx), dtype=np.uint8) 
        stack_zx = np.zeros((num_to_sample, 9, max_z, nx), dtype=np.uint8) 
        stack_zy = np.zeros((num_to_sample, 9, max_z, ny), dtype=np.uint8) 

        valid_count = 0
        for file in sampled_files:
            try:
                data_3d = self._load_gan(os.path.join(data_dir, file)) 
                if data_3d.size == 0 or data_3d.shape[0] == 0:
                    continue
                
                stack_xy[valid_count] = data_3d[random_z_slices, :, :]
                stack_zx[valid_count] = data_3d[:, random_y_slices, :].swapaxes(0, 1)
                stack_zy[valid_count] = data_3d[:, :, random_x_slices].transpose(2, 0, 1)
                valid_count += 1
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        stack_xy = stack_xy[:valid_count]
        stack_zx = stack_zx[:valid_count]
        stack_zy = stack_zy[:valid_count]
        
        self.plot_entropy_matrix_helper(stack_xy, random_z_slices, 'Z', 'X', 'Y', output_dir, data_dir, valid_count)
        self.plot_entropy_matrix_helper(stack_zx, random_y_slices, 'Y', 'X', 'Z', output_dir, data_dir, valid_count)
        self.plot_entropy_matrix_helper(stack_zy, random_x_slices, 'X', 'Y', 'Z', output_dir, data_dir, valid_count)

    def connectivity_and_pattern_analysis(self, target_val=1):
        """
        Executes the MPS and Connectivity analysis over the ensemble of models.
        """
        real_pattern_counter = Counter()
        gan_pattern_counter = Counter()
        
        all_real_blob_sizes = []
        all_gan_blob_sizes = []

        print(f"Processing {self.num_samples} Real samples...")
        for i in range(self.num_samples):
            # FIXED: Calling the internal method with self.
            real_data = self._load_real(self.real_files[i]) 
            
            # 1. Aggregate MPH Counts
            patterns, counts = self.get_pattern_counts(real_data, template_size=(3, 3, 3))
            for p, c in zip(patterns, counts):
                real_pattern_counter[tuple(p)] += c 
                
            # 2. Aggregate Connectivity
            blobs = self.analyze_connectivity(real_data, target_val)
            all_real_blob_sizes.extend(blobs)

        print(f"Processing {self.num_samples} GAN samples...")
        for i in range(self.num_samples):
            # FIXED: Calling the internal method with self.
            gan_data = self._load_gan(self.gan_files[i]) 
            
            # 1. Aggregate MPH Counts
            patterns, counts = self.get_pattern_counts(gan_data, template_size=(3, 3, 3))
            for p, c in zip(patterns, counts):
                gan_pattern_counter[tuple(p)] += c
                
            # 2. Aggregate Connectivity
            blobs = self.analyze_connectivity(gan_data, target_val)
            all_gan_blob_sizes.extend(blobs)
        
        # Convert Counter dictionaries back to probabilities
        total_real_patterns = sum(real_pattern_counter.values())
        total_gan_patterns = sum(gan_pattern_counter.values())
        
        print("\n--- Results ---")
        print(f"Total unique patterns found in Real ensemble: {len(real_pattern_counter)}")
        print(f"Total unique patterns found in GAN ensemble: {len(gan_pattern_counter)}")
        
        if len(all_real_blob_sizes) > 0 and len(all_gan_blob_sizes) > 0:
            print(f"\nReal ensemble max blob size: {max(all_real_blob_sizes)} voxels")
            print(f"GAN ensemble max blob size: {max(all_gan_blob_sizes)} voxels")
            
            print(f"Real ensemble median blob size: {np.median(all_real_blob_sizes)} voxels")
            print(f"GAN ensemble median blob size: {np.median(all_gan_blob_sizes)} voxels")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Initialize the class with your directory paths
    validator = PostProcessing1(
        real_path='datasets/training/training_dataset_upper_plain_delta_128/*.npy',
        gan_path='outputs/10000_training_samples/RUN_10000_samples_128xy_dataset_50_epochs_bs_32_corrected_nz/realizations/*.npy',
        num_samples=10
    )
    
    # Run the specific analysis method
    validator.connectivity_and_pattern_analysis(target_val=1)