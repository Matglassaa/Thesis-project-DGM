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

# Import custom utilities
from custom_plots import apply_custom_plotting_flavor

class PostProcessing1:
    """
    
    """
    def __init__(self, output_dir, real_path, gan_path, num_samples=10):
        """
        Initializes the class by loading ALL file paths.
        """
        self.output_dir = output_dir
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        # Load ALL available file paths
        self.real_files = sorted(glob.glob(real_path))
        self.gan_files = sorted(glob.glob(gan_path))
        
        # Subset limit for connectivity analysis
        self.num_samples = min(num_samples, len(self.real_files), len(self.gan_files))
        self.facies_mapping = {0: 1, 1: 4, 2: 8}

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
    
    def _create_colormap_and_legend():
        """
        Creates a ListedColormap and legend patches based on facies_properties.
        
        Returns:
            tuple: A tuple containing the custom colormap and a list of legend patches.
        """

        try:
            with open('scripts/gan_pipeline/core/facies_config.json','r') as f:
                facies_properties = json.load(f)
        except Exception as e:
            print(f"FileNotFound: Error loading facies configuration: {e}")
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
    
    def _plot_entropy_matrix_helper(self, slices_stack, slice_indices, axis_name, xlabel, ylabel, data_dir, num_files):
        """
        Helper function to calculate and plot the entropy matrices with dynamic titling.
        """
        num_realizations, _, dim_y, dim_x = slices_stack.shape
        facies_values = [1, 4, 8]
        
        vmin, vmax = 0, 1.6 # Max log2(3) ≈ 1.58
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, slice_val in enumerate(slice_indices):
            probs = np.zeros((len(facies_values), dim_y, dim_x))
            for i, f_val in enumerate(facies_values):
                probs[i] = np.sum(slices_stack[:, idx, :, :] == f_val, axis=0) / num_realizations

            entropy_map = entropy(probs, base=2, axis=0)
            im = axes[idx].imshow(entropy_map, cmap='magma', origin='lower', norm=norm)
            axes[idx].set_title(f"{axis_name}-Slice = {slice_val}", fontsize=12)
            axes[idx].set_xlabel(xlabel)
            axes[idx].set_ylabel(ylabel)

        # Formatting titles and colorbars
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label('Entropy (bits)', rotation=270, labelpad=15)

        # Dynamic naming as requested
        plane_names = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}
        plane = plane_names[axis_name]
        
        plt.suptitle(f"{plane} Plane Cell-Wise Entropy (Fixed Scale: {vmin}-{vmax} bits)\n"
                     f"({num_files} Realizations)", fontsize=16)

        # File naming logic
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        plot_path = os.path.join(self.output_dir, f"entropy_matrix_{plane}_{dataset_name}_{num_files}_samples.png")
        
        plt.savefig(plot_path, bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"Saved {plane} entropy plot to: {plot_path}")

    def plot_entropy(self):
        """
        Calculates and plots cell-wise entropy across ALL loaded GAN realizations.
        """
        num_total = len(self.gan_files)
        print(f"\n--- Generating Entropy Matrices ({num_total} Realizations) ---")
        
        if num_total == 0:
            print("No GAN files found.")
            return

        # Determine the data directory name for the output file naming
        data_dir = os.path.dirname(self.gan_files[0])

        # Load dimensions from the first realization
        sample_data = self._load_gan(self.gan_files[0])
        nz, ny, nx = sample_data.shape
        
        # Sample 9 consistent slices for visualization
        z_slices = sorted(random.sample(range(nz), 9))
        y_slices = sorted(random.sample(range(ny), 9))
        x_slices = sorted(random.sample(range(nx), 9))

        # Stacks for the entire ensemble
        stack_xy = np.zeros((num_total, 9, ny, nx), dtype=np.uint8) 
        stack_zx = np.zeros((num_total, 9, nz, nx), dtype=np.uint8) 
        stack_zy = np.zeros((num_total, 9, nz, ny), dtype=np.uint8) 

        for i, file_path in enumerate(self.gan_files):
            data_3d = self._load_gan(file_path)
            stack_xy[i] = data_3d[z_slices, :, :]
            stack_zx[i] = data_3d[:, y_slices, :].swapaxes(0, 1)
            stack_zy[i] = data_3d[:, :, x_slices].transpose(2, 0, 1)

        # Call helper with the dataset path and total count
        self._plot_entropy_matrix_helper(stack_xy, z_slices, 'Z', 'X', 'Y', data_dir, num_total)
        self._plot_entropy_matrix_helper(stack_zx, y_slices, 'Y', 'X', 'Z', data_dir, num_total)
        self._plot_entropy_matrix_helper(stack_zy, x_slices, 'X', 'Y', 'Z', data_dir, num_total)

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
    
    def plot_3d_pyvista(self, data_dir, output_dir, all_files):
        """
        Renders a 3D volumetric plot of a random sample using PyVista.
        
        Args:
            data_dir (str): Directory containing the .npz files.
            output_dir (str): Directory to save the resulting plot.
            all_files (list): List of all .npz file names.
        """
        print("\n--- Generating 3D PyVista Plot ---")
        try:
            import pyvista as pv
        except ImportError:
            print("Error: 'pyvista' is not installed. Skipping 3D plot.")
            return

        file_to_plot = random.choice(all_files)
        file_path = os.path.join(data_dir, file_to_plot)
        data_3d = self._load_gan(file_path)

        nz, ny, nx = data_3d.shape

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)

        grid.point_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

        custom_cmap, _ = self._create_colormap_and_legend()

        plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(grid, scalars='Facies', cmap=custom_cmap, clim=[0, 13])

        plot_path = os.path.join(output_dir, f"3d_plot_{os.path.splitext(file_to_plot)[0]}.png")
        plotter.show(screenshot=plot_path)
        print(f"Saved 3D plot to: {plot_path}")

if __name__ == "__main__":
    validator = PostProcessing1(
        output_dir='outputs/10000_training_samples/RUN_10000_samples_128xy_dataset_50_epochs_bs_64_val_size_020',
        real_path='datasets/training/training_dataset_upper_plain_delta_128/*.npy',
        gan_path='outputs/10000_training_samples/RUN_10000_samples_128xy_dataset_50_epochs_bs_64_val_size_020/realizations/*.npy',
        num_samples=10
    )

    # Apply custom plotting flavor for all plots generated in this script
    apply_custom_plotting_flavor()
    
    # Processes 10 samples (subset)
    validator.connectivity_and_pattern_analysis(target_val=1)
    
    # Processes ALL samples (full ensemble)
    validator.plot_entropy()