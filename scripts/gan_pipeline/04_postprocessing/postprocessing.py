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

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

# Import custom utilities
from gan_pipeline.core.custom_plots import apply_custom_plotting_flavor

class PostProcessing:
    """Handles spatial and statistical validation metrics for GAN-generated geological facies.

    This class provides tools to read, map, and analyze 3D categorical grid data
    from both ground-truth (real) datasets and Generative Adversarial Network
    outputs. It includes methods for spatial entropy mapping, volumetric facies 
    distributions, multi-point pattern frequency counts, connectivity analysis, 
    and 3D volumetric rendering.

    Attributes:
        output_dir (str): Destination path where generated plots and metrics are saved.
        script_path (pathlib.Path): Absolute path pointing to this script file.
        cwd (pathlib.Path): Absolute path pointing to the current working directory.
        real_files (list of str): Sorted file paths matching the real data directory pattern.
        gan_files (list of str): Sorted file paths matching the GAN outputs directory pattern.
        num_samples (int): The capped total of samples to evaluate for pattern/connectivity checks.
        facies_mapping (dict): Numerical mapping translating simplified IDs to evaluation labels.
    """
    def __init__(self, output_dir, real_path, gan_path, save_plots:bool, num_samples=10):
        """Initializes the class by loading ALL file paths.

        Args:
            output_dir (str): Directory where output files and figures will be saved.
            real_path (str): Glob-compatible path string matching the real .npy files.
            gan_path (str): Glob-compatible path string matching the GAN .npy/.npz files.
            num_samples (int, optional): Maximum number of samples to use for subset-heavy 
                analyses like connectivity. Defaults to 10.
        """ 
        self.output_dir = output_dir
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        # Load ALL available file paths
        self.real_files = sorted(glob.glob(str(real_path)))
        self.gan_files = sorted(glob.glob(str(gan_path)))
        
        # Subset limit for connectivity analysis
        self.num_samples = min(num_samples, len(self.real_files), len(self.gan_files))
        self.facies_mapping = {0: 1, 1: 4, 2: 8}
        self.save_plots = save_plots

    def _load_real(self, file):
        """Loads the facies array and maps real labels to representative facies [1, 4, 8].

        Args:
            file (str): File path to a ground-truth .npy grid.

        Returns:
            np.ndarray: A 3D numpy integer array with consolidated categorical labels (1, 4, or 8).
        """
        data = np.load(file)
        mapped_data = np.zeros_like(data)
        mapped_data[(data >= 0) & (data < 4)] = 1
        mapped_data[(data >= 4) & (data < 8)] = 4
        
        mapped_data[(data >= 8)] = 8
        
        return mapped_data
    
    def _load_gan(self, file):
        """Loads GAN output and standardizes it to a 3D categorical array."""
        if file.endswith('.npz'):
            with np.load(file) as data:
                raw_arr = data['facies'] if 'facies' in data else data[data.files[0]]
        else:
            raw_arr = np.load(file)

        if raw_arr.ndim == 4:
            # Fallback just in case you ever DO feed it a 4D array
            class_indices = np.argmax(raw_arr, axis=0) 
            
        elif raw_arr.ndim == 3:
            # Differentiate between floats (Tanh) and integers (Argmax/One-Hot)
            if raw_arr.min() < 0 or raw_arr.dtype in [np.float32, np.float64]:
                # Legacy handling: Continuous tanh data [-1, 1]
                class_indices = np.round(raw_arr + 1).astype(int)
                class_indices = np.clip(class_indices, 0, 2)
            else:
                # NEW: Data is already collapsed integers from the generation script
                class_indices = raw_arr.astype(int)
        else:
            raise ValueError(f"Unexpected array shape {raw_arr.shape} in file {file}")

        return self._map_indices_to_facies(class_indices)

    def _map_indices_to_facies(self, class_indices):
        """Maps 0-indexed categorical data to specific geological facies codes."""
        # Note: If you have 9 channels, mapping needs to support 9 values.
        # This is a fallback example; ideally, load this dynamically from facies_config.json
        
        if class_indices.max() < 3:
            mapping = np.array([1, 4, 8]) # For 3 channels
        else:
            # Example mapping for 9 channels - adjust to your real facies codes
            mapping = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 
            
        return mapping[class_indices]

    # def _load_gan_raw(self, file):
    #     """Loads the raw continuous GAN output array.

    #     Supports both standard text numpy array collections (`.npy`) and compressed 
    #     numpy archives (`.npz`).

    #     Args:
    #         file (str): File path to the GAN realization target.

    #     Returns:
    #         np.ndarray: Unprocessed continuous array data containing raw model outputs.
    #     """
    #     if file.endswith('.npz'):
    #         with np.load(file) as data:
    #             arr = data['facies'] if 'facies' in data else data[data.files[0]]
    #     else:
    #         arr = np.load(file)
    #     return arr

    # def _map_gan_to_facies(self, arr):
    #     """Maps continuous GAN labels [-1, 0, 1] to representative facies [1, 4, 8].

    #     Args:
    #         arr (np.ndarray): Continuous 3D numpy array from raw GAN tanh layers.

    #     Returns:
    #         np.ndarray: Categorical 3D integer array containing the mapped labels (1, 4, or 8).
    #     """
    #     class_indices = np.round(arr + 1).astype(int)
    #     class_indices = np.clip(class_indices, 0, 2)
    #     mapping = np.array([1, 4, 8])
    #     return mapping[class_indices]

    # def _load_gan(self, file):
    #     """Legacy wrapper to keep existing functions working.

    #     Args:
    #         file (str): File path to the GAN realization target.

    #     Returns:
    #         np.ndarray: Categorical 3D integer array mapped straight to standard IDs.
    #     """
    #     raw_arr = self._load_gan_raw(file)
    #     return self._map_gan_to_facies(raw_arr)
    
    def _create_colormap_and_legend(self):
        """Creates a ListedColormap and legend patches based on facies_properties.

        Returns:
            tuple: A two-element tuple containing:
                - custom_cmap (matplotlib.colors.ListedColormap or None): Ordered discrete color map 
                  if the configuration is found.
                - legend_patches (list of matplotlib.patches.Patch or None): Named legend handles 
                  for plots.
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
        """Extracts patterns and their raw counts from a single 3D array.

        Uses a sliding structural window approach to look at multi-point local configurations
        and group them into unique spatial configurations.

        Args:
            data_array (np.ndarray): Target 3D matrix representing a categorical grid.
            template_size (tuple of int, optional): Dimensions of the sliding structural voxel block. 
                Defaults to (3, 3, 3).

        Returns:
            tuple: A two-element tuple containing:
                - unique_patterns (np.ndarray): Two-dimensional array where each row represents 
                  a unique flattened voxel combination pattern.
                - counts (np.ndarray): One-dimensional array tracking total occurrences of each pattern.
        """
        windows = view_as_windows(data_array, template_size)
        
        n_patches = np.prod(windows.shape[:3])
        patch_size = np.prod(template_size)
        flat_windows = windows.reshape(n_patches, patch_size)
        
        unique_patterns, counts = np.unique(flat_windows, axis=0, return_counts=True)
        return unique_patterns, counts

    def analyze_connectivity(self, data_array, target_facies):
        """Finds connected 3D bodies of a specific facies and returns their volumes.

        Calculates disconnected component size volumes assuming full 26-neighbor 
        connectivity (all corners, faces, and edges touching).

        Args:
            data_array (np.ndarray): Target 3D array representing a categorical grid.
            target_facies (int): The discrete target ID (e.g., 1 for Channel) to isolate.

        Returns:
            np.ndarray: A sorted or raw 1D array where each value corresponds to the 
            total voxel count (volume) of an isolated geobody feature.
        """
        binary_mask = (data_array == target_facies).astype(int)
        structure = np.ones((3, 3, 3)) 
        
        labeled_array, num_features = label(binary_mask, structure=structure)
        unique_ids, blob_sizes = np.unique(labeled_array, return_counts=True)
        
        # Remove the background (ID 0)
        return blob_sizes[1:] 
    
    def _plot_entropy_matrix_helper(self, slices_stack, slice_indices, axis_name, xlabel, ylabel, data_dir, num_files):
        """Helper function to calculate and plot the entropy matrices with dynamic titling.

        Args:
            slices_stack (np.ndarray): Four-dimensional array tracking collected slice data 
                across the entire dataset ensemble. Shape: (num_realizations, 9, dim_y, dim_x).
            slice_indices (list of int): Grid index locations along the isolated axis.
            axis_name (str): Identifier label for the cross-section plane constraint ('X', 'Y', or 'Z').
            xlabel (str): Label applied to the x-axis for plotting.
            ylabel (str): Label applied to the y-axis for plotting.
            data_dir (str): Base file pathway used to dynamically label saved items.
            num_files (int): Total count of realizations evaluated for the figure.
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
        
        if self.save_plots:
            plt.savefig(plot_path, bbox_inches='tight', dpi=600)
            print(f"Saved {plane} entropy plot to: {plot_path}")
        plt.close(fig)

    def plot_facies_percentages(self, nc):
        """Plots the continuous and discrete percentage distributions across all GAN realizations.

        Builds a side-by-side subplot tracking raw value histograms (tanh spread) and 
        final mapped volume bars for individual rock/sediment types.

        Args:
            nc (int): Total expected number of structural classes to evaluate.
        """
        print(f"\n--- Generating Facies Distribution Plots across {len(self.gan_files)} Realizations ---")
        
        if not self.gan_files:
            print("Error: No GAN files found to process.")
            return

        target_facies = [1, 4, 8]
        facies_counts = {1: 0, 4: 0, 8: 0}
        total_voxels = 0
        
        # Setup bins for the raw continuous data histogram (assuming standard tanh output ~ -1.5 to 1.5)
        raw_bins = np.linspace(-1.5, 1.5, 100)
        raw_hist = np.zeros(len(raw_bins) - 1)

        # 1. Iterate over ALL files, accumulating data incrementally to save RAM
        for i, file_path in enumerate(self.gan_files):
            # Load raw data once
            raw_data = self._load_gan_raw(file_path)
            
            # Accumulate raw histogram
            counts, _ = np.histogram(raw_data, bins=raw_bins)
            raw_hist += counts
            
            # Map data and accumulate discrete counts
            mapped_data = self._map_gan_to_facies(raw_data)
            total_voxels += mapped_data.size
            
            for f_val in target_facies:
                facies_counts[f_val] += np.sum(mapped_data == f_val)

        # Calculate final percentages for discrete facies
        percentages = [(facies_counts[f] / total_voxels) * 100 for f in target_facies]

        # 2. Extract visualization metadata
        colors = []
        labels = []
        try:
            with open('scripts/gan_pipeline/core/facies_config.json', 'r') as f:
                facies_properties = json.load(f)
                
            sorted_items = sorted(facies_properties.items(), key=lambda x: x[1]['val'])
            for name, info in sorted_items:
                if info['val'] in target_facies:
                    colors.append(info['color'])
                    if info['val'] == 1: labels.append("Channel")
                    elif info['val'] == 4: labels.append("Crevasse Splay/Levee")
                    elif info['val'] == 8: labels.append("Floodplain")
        except Exception as e:
            print(f"Error loading JSON, using fallbacks. Error: {e}")
            colors = ['#f1970f', '#fffc65', '#33ff00']
            labels = ["Channel", "Crevasse Splay/Levee", "Floodplain"]

        # 3. Create a side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot A: Raw Continuous Distribution
        bin_centers = 0.5 * (raw_bins[1:] + raw_bins[:-1])
        # Normalize histogram to show density
        raw_hist_density = raw_hist / (np.sum(raw_hist) * np.diff(raw_bins)) 
        
        axes[0].plot(bin_centers, raw_hist_density, color='blue', lw=2)
        axes[0].fill_between(bin_centers, raw_hist_density, alpha=0.3, color='blue')
        axes[0].set_title('Raw GAN Output Distribution', fontsize=14, pad=10)
        axes[0].set_xlabel('Continuous Value (Raw)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Plot B: Clipped/Mapped Percentage Distribution
        bars = axes[1].bar(labels, percentages, color=colors, edgecolor='black', linewidth=1.2)
        axes[1].set_title(f'Mapped Facies Distribution ({len(self.gan_files)} Samples)', fontsize=14, pad=10)
        axes[1].set_ylabel('Volume Percentage (%)', fontsize=12)
        axes[1].set_ylim(0, 100)
        
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 2, 
                    f'{yval:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"facies_distributions_ensemble_{len(self.gan_files)}_samples.png")
        if self.save_plots:
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved facies distribution plot to: {plot_path}")
        plt.show(fig)
        

    def plot_entropy(self):
        """Calculates and plots cell-wise entropy across ALL loaded GAN realizations.

        Extracts 9 random but consistent slice coordinates on X, Y, and Z axes from the grid 
        dimensions, calculates information theory Shannon entropy on cell variations across 
        the ensemble, and compiles spatial variance matrix plots.
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
        """Executes Multi-Point Statistics (MPS) and Connectivity comparison across sets.

        Loops through real and fake sample pairs up to `self.num_samples` to gather structural
        pattern dictionaries and aggregate categorical object sizes, outputting comparison 
        statistics (max and median sizes) directly to terminal standard output.

        Args:
            target_val (int, optional): The focused discrete facies ID code to evaluate 
                for connected bodies. Defaults to 1 (Channel).
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
    
    def plot_3d_pyvista(self):
        """Renders a 3D volumetric plot of a random sample using PyVista.

        Generates 3 separate subplots in one saved composite image, where each window 
        applies threshold filtering to cleanly isolate exactly one facies indicator (1, 4, 8).

        Raises:
            FileNotFoundError: If the internal file list contains no valid target `.npy` components.
        """
        print("\n--- Generating 3D PyVista Plot ---")
        try:
            import pyvista as pv
        except ImportError:
            print("Error: 'pyvista' is not installed. Skipping 3D plot.")
            return

        npy_files = [f for f in self.gan_files if f.endswith(".npy")]

        if npy_files:
            random_file = random.choice(npy_files)
        else:
            raise FileNotFoundError("No .npy files found in self.gan_files list.")
        data_3d = self._load_gan(random_file)
        nz, ny, nx = data_3d.shape

        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)

        grid.cell_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

        facies_to_plot = [1, 4, 8]
        facies_titles = {1: "Channel", 4: "Crevasse Splay/Levee", 8: "Floodplain"}
        
        custom_cmap, _ = self._create_colormap_and_legend()

        plotter = pv.Plotter(shape=(1, 3), image_scale=4, off_screen=True, window_size=(7200, 2400))

        for i, f_val in enumerate(facies_to_plot):
            plotter.subplot(0, i)
            plotter.add_text(facies_titles[f_val], font_size=200, color='black', shadow=True)
            
            # Threshold to only include the current facies
            threshed = grid.threshold([f_val - 0.5, f_val + 0.5], scalars='Facies')

            if threshed.n_points > 0:
                plotter.add_mesh(
                    threshed, 
                    scalars = 'Facies',
                    cmap=custom_cmap, 
                    clim=[0, 13],
                    show_edges=False, 
                    show_scalar_bar=False,
                    ambient=0.2,
                    diffuse=0.8
                )

            #plotter.enable_depth_of_field()
            plotter.view_isometric()

        plot_path = os.path.join(self.output_dir, f"3d_plot_{os.path.splitext(os.path.basename(random_file))[0]}_facies.png")
        if self.save_plots:
            plotter.show(screenshot=plot_path)
            print(f"Saved 3D plot to: {plot_path}") 

if __name__ == "__main__":
    validator = PostProcessing(
        output_dir='outputs/20000_training_samples/RUN_20000_samples_50_epochs_bs_64_val_size_010_one_hot_all',
        real_path='datasets/training/training_dataset_upper_plain_delta_128/*.npy',
        gan_path='outputs/20000_training_samples/RUN_20000_samples_50_epochs_bs_64_val_size_010_one_hot_all/realizations/*.npy',
        num_samples=10
    )

    # Apply custom plotting flavor for all plots generated in this script
    apply_custom_plotting_flavor()

    # Plot the percentage distribution of facies for a random generated sample
    validator.plot_facies_percentages(nc=3)
    
    # Processes 10 samples (subset)
    validator.connectivity_and_pattern_analysis(target_val=1)
    
    # Processes ALL samples (full ensemble)
    validator.plot_entropy()

    # Generate 3D plot for a random sample
    validator.plot_3d_pyvista()