import os
import sys
import glob
import json
import random
import pathlib
import warnings
from pathlib import Path

# Matheatical libraries
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.ndimage import label
from scipy.spatial.distance import jensenshannon

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ML libraries
import torch
from sklearn.metrics import f1_score
from sklearn.manifold import MDS
from voxgan.models.metrics import MSSWD


# Miscelaneous
from tqdm import tqdm
from collections import Counter
from skimage.util import view_as_windows
from collections import Counter, defaultdict

# Adjust path based on your environment
scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))


class WellMismatch:
    """Quantifies the degree of mismatch between the 3D realizations and the well data.

    Calculates the Macro F1 score to evaluate how accurately the generated grid 
    captures the well data. Macro F1 is used to account for geological class 
    imbalance, giving equal weight to minority facies (e.g., channels).

    Attributes:
        data_files (list): Sorted list of file paths to the GAN realizations.
        well_data_path (str): Path to the well data file (e.g., CSV or Excel).
        well_coords (list): List of (Z, Y, X) tuples representing well locations.
        well_true_facies (list): List of true physical facies codes at the well coords.
    """
    def __init__(self, data_dir, well_data_path):
        self.data_files = sorted(glob.glob(str(data_dir)))
        if not self.data_files:
            print(f"Warning: No files found matching {data_dir}")
            
        self.well_data_path = well_data_path
        self.well_coords = []
        self.well_true_facies = []
        
        self._load_well_data()

    def _load_well_data(self):
        """Loads and maps the well data coordinates and true facies."""
        try:
            # Grid projection constants
            ORIGIN_E = 84337.0  
            ORIGIN_N = 445750.0 
            SPACING = 20.0
            
            tabs = ['DEL-GT-01', 'DEL-GT-02-S2']
            
            for tab in tabs:
                df = pd.read_excel(self.well_data_path, sheet_name=tab)
                df_32 = df.head(32) # Take first 32 meters (matches Z grid size)
                
                # Facies column might have a trailing space in some versions
                facies_col = 'Facies ' if 'Facies ' in df.columns else 'Facies'
                
                for i, row in df_32.iterrows():
                    z = i # meter by meter
                    y = int(np.round((row['GRID N'] - ORIGIN_N) / SPACING))
                    x = int(np.round((row['GRID E'] - ORIGIN_E) / SPACING))
                    
                    # Ensure within grid bounds (128x128)
                    if 0 <= x < 128 and 0 <= y < 128:
                        
                        # Extract raw facies and map it to 1, 4, or 8 to match the GAN output
                        raw_facies = row[facies_col]
                        
                        if 1 <= raw_facies <= 3: 
                            mapped_facies = 1
                        elif 4 <= raw_facies <= 7: 
                            mapped_facies = 4
                        elif 8 <= raw_facies <= 12: 
                            mapped_facies = 8
                        else:
                            mapped_facies = 1 # Fallback
                            
                        self.well_coords.append((z, y, x))
                        self.well_true_facies.append(mapped_facies)
                
            print(f"Loaded {len(self.well_coords)} well conditioning points.")
                
        except Exception as e:
            print(f"Error loading well data: {e}")

    def _load_and_map_realization(self, file_path):
        """Loads a single realization and maps it to physical facies codes."""
        if file_path.endswith('.npz'):
            with np.load(file_path) as data:
                raw_arr = data['facies'] if 'facies' in data else data[data.files[0]]
        else:
            raw_arr = np.load(file_path)
            
        # Convert raw network output to standard 3D indices
        if raw_arr.ndim == 4:
            class_indices = np.argmax(raw_arr, axis=0) 
        elif raw_arr.ndim == 3:
            class_indices = np.round(raw_arr).astype(int)
        else:
            raise ValueError(f"Unexpected array shape {raw_arr.shape} in {file_path}")

        # Map to physical facies: 1, 4, 8
        mapping = np.array([1, 4, 8])
        class_indices = np.clip(class_indices, 0, 2)
        return mapping[class_indices]

    def compute_mismatch(self):
        """Computes the Macro F1 score for all loaded realizations against well data.

        Returns:
            pd.DataFrame: A dataframe containing the F1 Macro score for each file.
        """
        if not self.well_coords:
            print("No well coordinates loaded. Cannot compute mismatch.")
            return None

        results = []
        
        for file_path in self.data_files:
            try:
                grid = self._load_and_map_realization(file_path)
                
                y_pred = []
                y_true_valid = []
                
                # Extract predicted values at exactly the well coordinates
                for (z, y, x), true_facies in zip(self.well_coords, self.well_true_facies):
                    # Safety check to ensure well coords fall inside the 3D grid boundaries
                    if 0 <= z < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= x < grid.shape[2]:
                        y_pred.append(grid[z, y, x])
                        y_true_valid.append(true_facies)
                    
                # Calculate the Macro F1 Score
                macro_f1 = f1_score(y_true_valid, y_pred, average='macro')
                
                results.append({
                    'Realization': os.path.basename(file_path),
                    'Macro_F1_Score': macro_f1
                })
                
            except Exception as e:
                print(f"Error processing {file_path} for mismatch: {e}")
                
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            mean_f1 = df_results['Macro_F1_Score'].mean()
            print(f"\n--- Well Data Mismatch ---")
            print(f"Evaluated {len(results)} realizations.")
            print(f"Average Macro F1 Score: {mean_f1:.4f}")
        
        return df_results

    def plot_vpc(self, num_realizations=10, save_plot=False, output_dir='outputs'):
        """Calculates and plots the Vertical Proportion Curve (VPC) comparing Well vs GAN data."""
        print(f"\n--- Generating Vertical Proportion Curves (VPC) ---")
        if not self.well_coords:
            print("No well data loaded. Cannot compute VPC.")
            return

        # 1. Set up depth parameters
        # Based on your well loading logic, we evaluate 32 meters
        max_z = 32
        facies_codes = [1, 4, 8]
        facies_colors = {1: '#f1970f', 4: '#fffc65', 8: '#33ff00'}
        facies_labels = {1: 'Sand body deposits', 4: 'Crevasse splay & Levee deposits', 8: 'Clay deposits'}

        # 2. Calculate Well VPC
        well_vpc = {f: np.zeros(max_z) for f in facies_codes}
        well_counts_per_z = np.zeros(max_z)

        # Tally the facies at each depth index from the exact well locations
        for (z, y, x), f_val in zip(self.well_coords, self.well_true_facies):
            if z < max_z:
                well_vpc[f_val][z] += 1
                well_counts_per_z[z] += 1

        # Normalize counts into proportions (0.0 to 1.0)
        for f in facies_codes:
            # Avoid division by zero where we have no well data
            safe_counts = np.where(well_counts_per_z == 0, 1, well_counts_per_z)
            well_vpc[f] = well_vpc[f] / safe_counts
            # Set to NaN so matplotlib doesn't plot zeros where data is missing
            well_vpc[f][well_counts_per_z == 0] = np.nan 

        # 3. Calculate GAN VPC
        gan_vpc = {f: np.zeros(max_z) for f in facies_codes}
        files_to_process = self.data_files[:num_realizations]
        
        print(f"Processing {len(files_to_process)} GAN realizations for global VPC...")
        for file_path in files_to_process:
            grid = self._load_and_map_realization(file_path)
            z_dim, y_dim, x_dim = grid.shape
            cells_per_slice = y_dim * x_dim
            
            for f in facies_codes:
                # Count occurrences of facies 'f' in every Z-slice, divide by total area
                proportions = np.sum(grid == f, axis=(1, 2)) / cells_per_slice
                gan_vpc[f] += proportions[:max_z]
                
        # Average the GAN proportions across all processed realizations
        for f in facies_codes:
            gan_vpc[f] /= len(files_to_process)

        # 4. Render the Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        z_array = np.arange(max_z)

        # Plot Well Data
        for f in facies_codes:
            axes[0].plot(well_vpc[f], z_array, color=facies_colors[f], 
                         label=facies_labels[f], linewidth=2.5, marker='o', markersize=4)
        
        axes[0].set_title('Well Data VPC (Local)', fontsize=14)
        axes[0].set_ylabel('Depth (Z-index)', fontsize=12)
        axes[0].set_xlabel('Proportion', fontsize=12)
        axes[0].invert_yaxis() # Depth typically goes down
        axes[0].set_xlim(0, 1)
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot GAN Data
        for f in facies_codes:
            axes[1].plot(gan_vpc[f], z_array, color=facies_colors[f], 
                         linewidth=2.5)
            # Add a subtle fill for better readability
            axes[1].fill_betweenx(z_array, 0, gan_vpc[f], color=facies_colors[f], alpha=0.1)

        axes[1].set_title(f'GAN Data VPC (Global Mean of {len(files_to_process)} samples)', fontsize=14)
        axes[1].set_xlabel('Proportion', fontsize=12)
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Shared Legend
        handles = [plt.Line2D([0], [0], color=facies_colors[f], lw=4) for f in facies_codes]
        fig.legend(handles, [facies_labels[f] for f in facies_codes], 
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=12)

        plt.suptitle("Vertical Proportion Curve: Well Conditioning vs. GAN Generation", fontsize=16)
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"vpc_comparison_{num_realizations}_samples.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved VPC plot to: {plot_path}")

        plt.show()


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
    
    def __init__(self, name, output_dir, real_path, gan_path):
        """Initializes the class and locates files, but defers loading to save memory.

        Args:
            output_dir (str): Directory to save outputs (e.g., 'outputs/metrics').
            real_path (str): Glob pattern matching real data files (e.g., 'datasets/*.npy').
            gan_path (str): Glob pattern matching GAN data files (e.g., 'outputs/*.npy').
        """ 
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        self.real_files = sorted(glob.glob(str(real_path)))
        self.gan_files = sorted(glob.glob(str(gan_path)))
        
        self.real_data = []
        self.gan_data = []
        
        self.facies_mapping = {0: 1, 1: 4, 2: 8}
    
    def load_flumy_samples(self, limit=100):
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
        """Extracts and counts Multiple Point Statistics (MPS) patterns via sliding window."""
        windows = view_as_windows(data_array, template_size)
        n_patches = np.prod(windows.shape[:3])
        patch_size = np.prod(template_size)
        flat_windows = windows.reshape(n_patches, patch_size)
        return np.unique(flat_windows, axis=0, return_counts=True)

    def analyze_connectivity(self, data_array, target_facies):
        """Calculates 3D structural connectivity sizes for a specific facies."""
        binary_mask = (data_array == target_facies).astype(int)
        structure = np.ones((3, 3, 3)) 
        labeled_array, _ = label(binary_mask, structure=structure)
        unique_ids, blob_sizes = np.unique(labeled_array, return_counts=True)
        return blob_sizes[1:] 

    def connectivity_and_pattern_analysis(self, facies_list=None, sample_limit=10):
        """Executes MPS and Connectivity comparisons for ALL facies using loaded data.

        Args:
            facies_list (list, optional): Specific facies to analyze. If None, it will 
                automatically detect all unique facies in the real dataset.
            sample_limit (int, optional): Maximum number of samples to process. 
                Defaults to 10. Pass None to process all.
                
        Returns:
            tuple: (DataFrame of connectivity stats, Real Pattern Counter, GAN Pattern Counter)
        """
        load_limit = sample_limit if sample_limit is not None else 10
        if getattr(self, 'real_data', None) is None or getattr(self, 'gan_data', None) is None:
            print(f"Warning: Data not loaded. Auto-loading up to {load_limit} samples...")
            self.load_flumy_samples(limit=load_limit)
            self.load_gan_samples(limit=load_limit)

        real_to_process = self.real_data[:sample_limit] if sample_limit else self.real_data
        gan_to_process = self.gan_data[:sample_limit] if sample_limit else self.gan_data

        # Automatically detect unique facies if not provided
        if facies_list is None:
            facies_list = np.unique(real_to_process[0]).tolist()
            print(f"Auto-detected facies: {facies_list}")

        real_pattern_counter, gan_pattern_counter = Counter(), Counter()
        
        # Use defaultdict to dynamically store lists of blob sizes for every facies
        all_real_blobs = defaultdict(list)
        all_gan_blobs = defaultdict(list)

        # --- Process Real Data ---
        print(f"Processing {len(real_to_process)} Real samples...")
        for data in real_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                real_pattern_counter[tuple(p)] += c 
                
            # Check connectivity for EVERY facies
            for f_val in facies_list:
                all_real_blobs[f_val].extend(self.analyze_connectivity(data, f_val))

        # --- Process GAN Data ---
        print(f"Processing {len(gan_to_process)} GAN samples...")
        for data in gan_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                gan_pattern_counter[tuple(p)] += c
                
            # Check connectivity for EVERY facies
            for f_val in facies_list:
                all_gan_blobs[f_val].extend(self.analyze_connectivity(data, f_val))
        
        # --- Output Pattern Results ---
        print("\n" + "="*40)
        print(" MULTIPLE POINT STATISTICS (MPS)")
        print("="*40)
        print(f"Total unique patterns (Real): {len(real_pattern_counter)}")
        print(f"Total unique patterns (GAN):  {len(gan_pattern_counter)}")
        
        # --- Compile Connectivity Results into a DataFrame ---
        stats_data = []
        for f_val in facies_list:
            r_blobs = all_real_blobs[f_val]
            g_blobs = all_gan_blobs[f_val]
            
            stats_data.append({
                'Facies': f_val,
                'Real_Max_Blob': max(r_blobs) if r_blobs else 0,
                'GAN_Max_Blob': max(g_blobs) if g_blobs else 0,
                'Real_Median': np.median(r_blobs) if r_blobs else 0,
                'GAN_Median': np.median(g_blobs) if g_blobs else 0,
                'Real_Mean': np.mean(r_blobs) if r_blobs else 0,
                'GAN_Mean': np.mean(g_blobs) if g_blobs else 0
            })
            
        df_stats = pd.DataFrame(stats_data)
        
        # Format the float columns for cleaner console output
        df_stats['Real_Mean'] = df_stats['Real_Mean'].round(2)
        df_stats['GAN_Mean'] = df_stats['GAN_Mean'].round(2)
        
        print("\n" + "="*40)
        print(" MACRO-CONNECTIVITY STATISTICS")
        print("="*40)
        print(df_stats.to_string(index=False))
        print("\n")

        return df_stats, real_pattern_counter, gan_pattern_counter

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
            self.load_flumy_samples()
            
        print(f"\n--- Generating Facies Distribution (Mode: {mode.upper()}) ---")

        target_facies = [1, 4, 8]
        labels = ["Sand body deposits", "Crevasse Splay/Levee", "Floodplain"]
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
        
        # WE MUST LOAD REAL DATA: Even if plotting 'gan', we need 'real' data to compute the baseline H_max
        if not self.real_data:
            self.load_flumy_samples()
            
        target_data = self.gan_data if data_source == 'gan' else self.real_data
        num_total = len(target_data)
        axes_to_plot = [axis] if axis else ['Z', 'Y', 'X']

        # NEW LOGIC: Calculate theoretical max entropy strictly from the REAL dataset
        real_global_probs = self._get_global_proportions(self.real_data)
        h_max = self._calculate_h_max(real_global_probs) * 1.3
        
        print(f"\n--- Generating {data_source.upper()} Normalized Entropy Matrices ---")
        print(f"Real Dataset Baseline Proportions: {[round(p, 4) for p in real_global_probs]}")
        print(f"Target Baseline H_max:  {h_max:.4f} bits")

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

    def compute_spatial_delentropy(self, data_array):
        """Calculates the 3D spatial Delentropy of a single geological volume.
        
        This metric accounts for spatial structure by calculating the Shannon 
        entropy of the joint probability distribution of the 3D gradient vector field.
        
        Args:
            data_array (numpy.ndarray): 3D geological volume (Z, Y, X).
            
        Returns:
            float: The computed spatial delentropy in bits.
        """
        # 1. Compute discrete finite differences (gradients) along Z, Y, and X axes
        # We use np.diff and pad to maintain the original shape of the array
        dz = np.pad(np.diff(data_array, axis=0), ((0, 1), (0, 0), (0, 0)), mode='edge')
        dy = np.pad(np.diff(data_array, axis=1), ((0, 0), (0, 1), (0, 0)), mode='edge')
        dx = np.pad(np.diff(data_array, axis=2), ((0, 0), (0, 0), (0, 1)), mode='edge')

        # 2. Flatten and stack to create the phase space of gradient vectors (dz, dy, dx)
        gradients = np.vstack([dz.flatten(), dy.flatten(), dx.flatten()]).T

        # 3. Calculate Deldensity (the joint 3D histogram of gradient occurrences)
        _, counts = np.unique(gradients, axis=0, return_counts=True)
        
        # 4. Convert counts to probabilities
        probabilities = counts / counts.sum()

        # 5. Calculate the Shannon joint entropy of the gradient field
        # The paper halves the 2D entropy to account for Papoulis' sampling redundancy, 
        # but for comparative metrics between datasets, the raw joint entropy is an excellent structural indicator.
        spatial_entropy = entropy(probabilities, base=2)
        
        # We divide by 2 following the parsimony principle outlined for Delentropy 
        # to halve the vector data rate using generalized sampling expansion.
        delentropy = spatial_entropy / 2.0 

        return delentropy

    def compare_structural_delentropy(self):
        """Evaluates and compares the spatial Delentropy of Real vs GAN datasets."""
        if not self.gan_data: self.load_gan_samples()
        if not self.real_data: seflumy()

        print("\n--- Computing 3D Spatial Delentropy (Structural Complexity) ---")
        
        real_entropies = [self.compute_spatial_delentropy(arr) for arr in self.real_data]
        gan_entropies = [self.compute_spatial_delentropy(arr) for arr in self.gan_data]

        real_mean = np.mean(real_entropies)
        gan_mean = np.mean(gan_entropies)
        
        print(f"Real Data Mean Delentropy: {real_mean:.4f} bits")
        print(f"GAN Data Mean Delentropy:  {gan_mean:.4f} bits")
        print(f"Difference (GAN - Real):   {gan_mean - real_mean:.4f} bits")
        
        if abs(gan_mean - real_mean) < 0.1:
            print("-> The GAN exhibits excellent spatial structural fidelity compared to the real data.")
        else:
            print("-> A notable difference in spatial entropy suggests structural deviations (e.g., over-smoothing or excessive noise).")
            
        return {'real_delentropy_mean': real_mean, 'gan_delentropy_mean': gan_mean}

    def plot_local_delentropy_map(self, data_source='gan', slice_idx=None, window_size=5):
        """Generates and plots a 2D spatial map of local Delentropy using a sliding window.
        
        Args:
            data_source (str): 'gan' or 'real'.
            slice_idx (int, optional): The Z-slice index to plot. If None, picks the middle slice.
            window_size (int, optional): The size of the sliding window (must be odd, e.g., 5, 7, 9).
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number (e.g., 3, 5, 7).")

        # 1. Load data and select a random realization
        if data_source == 'gan' and not self.gan_data: self.load_gan_samples()
        if data_source == 'real' and not self.real_data: seflumy()
        
        target_data = self.gan_data if data_source == 'gan' else self.real_data
        volume = target_data[random.randint(0, len(target_data) - 1)]
        
        # Select middle Z-slice if none provided
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
            
        slice_2d = volume[slice_idx, :, :]
        ny, nx = slice_2d.shape
        
        # 2. Pad the slice so the sliding window can reach the edges
        pad_w = window_size // 2
        padded_slice = np.pad(slice_2d, pad_w, mode='edge')
        
        # 3. Extract all local patches using a sliding window
        windows = view_as_windows(padded_slice, (window_size, window_size))
        
        # Prepare the output heatmap array
        local_entropy_map = np.zeros((ny, nx))
        
        print(f"Calculating local Delentropy map (Window: {window_size}x{window_size})...")
        
        # 4. Calculate Delentropy for each local patch
        for y in range(ny):
            for x in range(nx):
                patch = windows[y, x]
                
                # Compute discrete gradients (dy, dx) within the patch
                # We trim by [:, :-1] and [:-1, :] so both gradient arrays have shape (w-1, w-1)
                dy = np.diff(patch, axis=0)[:, :-1].flatten()
                dx = np.diff(patch, axis=1)[:-1, :].flatten()
                
                gradients = np.vstack([dy, dx]).T
                
                # Build local Deldensity and compute Shannon entropy
                _, counts = np.unique(gradients, axis=0, return_counts=True)
                probs = counts / counts.sum()
                
                # Halve the entropy for parsimony (as defined in the Delentropy literature)
                local_entropy_map[y, x] = entropy(probs, base=2) / 2.0

        # 5. Visual Rendering
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot A: Original Geological Facies
        custom_cmap, legend_patches = self._create_colormap_and_legend()
        if custom_cmap:
            # Map physical codes (1, 4, 8) to (0, 1, 2) for the colormap index
            display_slice = np.zeros_like(slice_2d)
            display_slice[slice_2d == 4] = 1
            display_slice[slice_2d == 8] = 2
            im0 = axes[0].imshow(display_slice, cmap=custom_cmap, origin='lower')
            axes[0].legend(handles=legend_patches, loc='upper right')
        else:
            im0 = axes[0].imshow(slice_2d, cmap='viridis', origin='lower')
            
        axes[0].set_title(f"Original Facies (Z-Slice {slice_idx})")
        
        # Plot B: Local Delentropy Heatmap
        im1 = axes[1].imshow(local_entropy_map, cmap='magma', origin='lower')
        axes[1].set_title(f"Local Delentropy Map ({window_size}x{window_size} Window)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Local Spatial Entropy (bits)")
        
        plt.suptitle(f"{data_source.upper()} Data - Spatial Structural Mapping", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def compute_slice_metrics(self, axis='Z', plot=False):
        """Computes slice-wise aggregate scalar metrics comparing GAN to Real distributions.
        
        Evaluates Mean Normalized Entropy and Jensen-Shannon Divergence (JSD) 
        across EVERY slice in the specified axis.

        Args:
            axis (str, optional): Target axis to slice ('X', 'Y', 'Z'). Defaults to 'Z'.
            plot (bool, optional): If True, displays a distribution plot of the metrics.

        Returns:
            tuple: 
                - pandas.DataFrame: Contains Slice Index, Real/GAN Entropy, and JSD for ALL slices.
                - dict: Mean and Standard Deviation for Entropy and JSD across the axis.
        """
        if not self.gan_data: self.load_gan_samples()
        if not self.real_data: seflumy()
            
        print(f"\n--- Computing Scalar Metrics for ALL {axis}-Axis Slices ---")
        
        real_global_probs = self._get_global_proportions(self.real_data)
        h_max = self._calculate_h_max(real_global_probs)
        
        facies_values = [1, 4, 8]
        num_real_samples = len(self.real_data)
        num_gan_samples = len(self.gan_data)
        
        nz, ny, nx = self.real_data[0].shape
        dims = {'Z': nz, 'Y': ny, 'X': nx}
        
        # NEW: Evaluate every single slice along the chosen axis
        slice_indices = range(dims[axis])
        results = []

        # Move warnings filter outside the loop for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            with tqdm(total=len(slice_indices), desc=f"JS-entropy 2D ({axis}-axis)") as pbar:
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
                        
                    # Normalize Real Entropy against the Real H_max
                    entropy_real = np.nanmean(entropy(p_real, base=2, axis=0))
                    if h_max > 0: entropy_real /= h_max
                    
                    # Normalize GAN Entropy against the Real H_max
                    entropy_gan = np.nanmean(entropy(p_gan, base=2, axis=0))
                    if h_max > 0: entropy_gan /= h_max
                    
                    # FIX: Added axis=0 to properly calculate pixel-wise distance!
                    js_distance = jensenshannon(p_real, p_gan, base=2, axis=0)
                    mean_jsd = np.nanmean(js_distance ** 2)
                    
                    results.append({
                        'Axis': axis,
                        'Slice_Index': slice_idx,
                        'Real_Norm_Entropy': entropy_real,
                        'GAN_Norm_Entropy': entropy_gan,
                        'JSD_Real_vs_GAN': mean_jsd
                    })
                    pbar.update(1)
                
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate Mean and Std Deviation
        summary_stats = {
            'Real_Entropy_Mean': df_results['Real_Norm_Entropy'].mean(),
            'Real_Entropy_Std': df_results['Real_Norm_Entropy'].std(),
            'GAN_Entropy_Mean': df_results['GAN_Norm_Entropy'].mean(),
            'GAN_Entropy_Std': df_results['GAN_Norm_Entropy'].std(),
            'JSD_Mean': df_results['JSD_Real_vs_GAN'].mean(),
            'JSD_Std': df_results['JSD_Real_vs_GAN'].std(),
        }

        print(f"\n--- Summary Statistics ({axis}-Axis) ---")
        print(f"Real Entropy: {summary_stats['Real_Entropy_Mean']:.4f} ± {summary_stats['Real_Entropy_Std']:.4f}")
        print(f"GAN Entropy:  {summary_stats['GAN_Entropy_Mean']:.4f} ± {summary_stats['GAN_Entropy_Std']:.4f}")
        print(f"JSD (Spatial): {summary_stats['JSD_Mean']:.4f} ± {summary_stats['JSD_Std']:.4f}\n")
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"slice_metrics_normalized_{axis}.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Saved full slice metrics to: {csv_path}")

        # Optional Plotting
        if plot:
            self._plot_metric_distributions(df_results, axis)
            
        return df_results, summary_stats
    
    def plot_2d_slices(self, data_source='gan', num_samples=1, num_slices=1, axis=None, show_plot=True, save_plot=False):
        """Visualizes 2D cross-sections (Horizontal and Vertical) of 3D realizations.
        
        Args:
            data_source (str): Target dataset to evaluate ('gan' or 'real').
            num_samples (int): Number of realizations to plot.
            num_slices (int): Number of slices to extract per realization (1, 3, or 9).
            axis (str, optional): Specific axis to slice ('X', 'Y', 'Z'). If None, plots all 3.
            show_plot (bool): If True, displays the plot interactively.
            save_plot (bool): If True, saves the plot to the output directory.
        """
        valid_slices = [1, 3, 9]
        if num_slices not in valid_slices:
            raise ValueError(f"num_slices must be 1, 3, or 9. Received: {num_slices}")
            
        if axis and axis.upper() not in ['X', 'Y', 'Z']:
            raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
            
        axis = axis.upper() if axis else None

        # Ensure data is loaded
        if data_source == 'gan' and not self.gan_data: 
            self.load_gan_samples(limit=num_samples)
        if data_source == 'real' and not self.real_data: 
            seflumy(limit=num_samples)
            
        target_data = self.gan_data if data_source == 'gan' else self.real_data
        files_list = self.gan_files if data_source == 'gan' else self.real_files
        
        plot_limit = min(num_samples, len(target_data))
        print(f"\n--- Generating 2D Slices ({data_source.upper()} Data | {num_slices} Slices | {plot_limit} Samples) ---")

        # 1. Match colors exactly to PyVista 3D settings
        colors = ['#f1970f', '#fffc65', '#33ff00']
        cmap = ListedColormap(colors)
        labels = ["Channel", "Crevasse Splay/Levee", "Floodplain"]
        legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]

        for idx in range(plot_limit):
            data_3d = target_data[idx]
            
            # 2. Map physical codes (1, 4, 8) to (0, 1, 2) for the colormap
            display_data = np.zeros_like(data_3d)
            display_data[data_3d == 1] = 0  
            display_data[data_3d == 4] = 1  
            display_data[data_3d == 8] = 2  

            depth, height, width = display_data.shape
            
            # 3. Calculate evenly spaced slices across the dimensions
            z_idx = np.linspace(0, depth - 1, num_slices, dtype=int)
            y_idx = np.linspace(0, height - 1, num_slices, dtype=int)
            x_idx = np.linspace(0, width - 1, num_slices, dtype=int)
            
            filename = os.path.basename(files_list[idx]) if idx < len(files_list) else f"Sample_{idx}"

            if axis is None:
                fig, axes = plt.subplots(num_slices, 3, figsize=(18, 5 * num_slices))
                
                # Expand dimensions if only 1 slice to keep the axes[row, col] indexing consistent
                if num_slices == 1:
                    axes = np.expand_dims(axes, axis=0) 
                
                for i in range(num_slices):
                    # Horizontal Slice (Z)
                    axes[i, 0].imshow(display_data[z_idx[i], :, :], cmap=cmap, origin='lower', vmin=0, vmax=2)
                    axes[i, 0].set_title(f"Horizontal Slice (Z={z_idx[i]})")
                    axes[i, 0].set_xlabel("X (Width)")
                    axes[i, 0].set_ylabel("Y (Height)")

                    # Vertical Section (Y)
                    axes[i, 1].imshow(display_data[:, y_idx[i], :], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=2) 
                    axes[i, 1].set_title(f"Vertical Section (Y={y_idx[i]})")
                    axes[i, 1].set_xlabel("X (Width)")
                    axes[i, 1].set_ylabel("Z (Depth)")

                    # Vertical Section (X)
                    axes[i, 2].imshow(display_data[:, :, x_idx[i]], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=2)
                    axes[i, 2].set_title(f"Vertical Section (X={x_idx[i]})")
                    axes[i, 2].set_xlabel("Y (Height)")
                    axes[i, 2].set_ylabel("Z (Depth)")

            else:
                if num_slices == 1:
                    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
                    axes_list = [axes]
                elif num_slices == 3:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    axes_list = axes.flatten()
                else: # 9 slices
                    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
                    axes_list = axes.flatten()

                for i in range(num_slices):
                    ax = axes_list[i]
                    if axis == 'Z':
                        ax.imshow(display_data[z_idx[i], :, :], cmap=cmap, origin='lower', vmin=0, vmax=2)
                        ax.set_title(f"Horizontal Slice (Z={z_idx[i]})")
                        ax.set_xlabel("X (Width)")
                        ax.set_ylabel("Y (Height)")
                    elif axis == 'Y':
                        ax.imshow(display_data[:, y_idx[i], :], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=2)
                        ax.set_title(f"Vertical Section (Y={y_idx[i]})")
                        ax.set_xlabel("X (Width)")
                        ax.set_ylabel("Z (Depth)")
                    elif axis == 'X':
                        ax.imshow(display_data[:, :, x_idx[i]], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=2)
                        ax.set_title(f"Vertical Section (X={x_idx[i]})")
                        ax.set_xlabel("Y (Height)")
                        ax.set_ylabel("Z (Depth)")

            # 4. Final touches: Legend, title, layout
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=12)
            plt.suptitle(f"Realization: {filename} | Shape: {data_3d.shape}", fontsize=16, fontweight='bold', y=1.02)
            #plt.tight_layout()
            
            # 5. Output handling
            if save_plot:
                ax_label = axis if axis else "All_Axes"
                plot_path = os.path.join(self.output_dir, f"2d_slices_{data_source}_{ax_label}_{num_slices}slices_{filename.replace('.npy', '.png')}")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"Saved 2D slices to: {plot_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

    def plot_3d_pyvista(self, data_source='gan', mode='separate', target_filename=None, target_facies=None, show_legend=True, show_plot=True, save_plot=False):
        """Renders an interactive or static 3D volumetric plot of a geological sample.

        Args:
            data_source (str, optional): Target dataset to sample ('gan' or 'real'). Defaults to 'gan'.
            mode (str, optional): Visual layout mode ('separate' for subplots, or 'combined' for 
                a single overlaid volume). Defaults to 'separate'.
            target_filename (str, optional): Specific filename to load (e.g., 'realization_01.npy'). 
                If None, a random sample is chosen from loaded data.
            target_facies (int or list, optional): Specific facies code(s) to isolate (e.g., 1, 4, or 8). 
                If None, plots all standard facies.
            show_legend (bool, optional): Toggles legend in combined mode. Defaults to True.
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
            
        try:
            import pyvista as pv
            pv.set_jupyter_backend('static') 
        except ImportError:
            print("Error: 'pyvista' is not installed. Skipping 3D plot.")
            return

        print(f"\n--- Generating 3D PyVista Plot ({data_source.upper()} Data | Mode: {mode.upper()}) ---")

        # 1. Determine which 3D array to plot (Specific File vs Random)
        plot_identifier = ""
        if target_filename:
            file_list = self.gan_files if data_source == 'gan' else self.real_files
            matched_file = next((f for f in file_list if target_filename in f), None)
            
            if matched_file:
                print(f"Loading specific file: {matched_file}")
                data_3d = self._load_gan(matched_file) if data_source == 'gan' else self._load_real(matched_file)
                plot_identifier = target_filename.split('.')[0]
            else:
                print(f"Warning: '{target_filename}' not found. Falling back to random sample.")
                target_filename = None # Trigger fallback

        if not target_filename:
            # Ensure data is loaded for random sampling
            if data_source == 'gan' and not self.gan_data:
                self.load_gan_samples()
            elif data_source == 'real' and not self.real_data:
                seflumy()
                
            target_data = self.gan_data if data_source == 'gan' else self.real_data
            random_idx = random.randint(0, len(target_data) - 1)
            data_3d = target_data[random_idx]
            plot_identifier = f"sample_{random_idx}"

        nz, ny, nx = data_3d.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.cell_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

        facies_colors = {1: '#f1970f', 4: '#fffc65', 8: '#33ff00'}
        facies_titles = {1: "Channel", 4: "Crevasse Splay/Levee", 8: "Floodplain"}
        
        # 2. Determine which facies to plot/isolate
        if target_facies is None:
            facies_to_plot = [1, 4, 8]
        elif isinstance(target_facies, int):
            facies_to_plot = [target_facies]
        else:
            facies_to_plot = target_facies
            
        # Automatically switch to 'combined' if isolating only one facies to avoid redundant subplots
        if len(facies_to_plot) == 1 and mode == 'separate':
            mode = 'combined'

        # 3. Setup Plotter Canvas
        if mode == 'separate':
            num_subplots = len(facies_to_plot)
            plotter = pv.Plotter(shape=(1, num_subplots), image_scale=4, off_screen=save_plot and not show_plot, window_size=(2400 * num_subplots, 2400))
        else:
            plotter = pv.Plotter(shape=(1, 1), image_scale=4, off_screen=save_plot and not show_plot, window_size=(2400, 2400))
            
        # 4. Render Meshes
        for i, f_val in enumerate(facies_to_plot):
            if f_val not in facies_colors:
                continue # Skip unrecognized facies
                
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

        if (mode == 'combined') and (show_legend == True) and len(facies_to_plot) > 1:
            plotter.add_legend(bcolor='grey', face=None, size=(0.2, 0.2))
        
        if mode == 'combined':
            plotter.view_isometric()

        # 5. Output Handling
        suffix = f"_{'-'.join(map(str, facies_to_plot))}" if target_facies else ""
        plot_path = os.path.join(self.output_dir, f"3d_plot_{data_source}_{plot_identifier}_{mode}{suffix}.png")
        
        if save_plot:
            plotter.show(screenshot=plot_path)
            print(f"Saved 3D plot to: {plot_path}") 
        elif show_plot:
            plotter.show()
        else:
            plotter.close()

class DistributionEvaluator:
    """Evaluates geological distributions using Multi-Scale Sliced Wasserstein Distance (MS-SWD).
    
    Computes pairwise MS-SWD between Real and GAN samples, and reduces the distance 
    matrix to a 2D embedding using Multidimensional Scaling (MDS) for visualization.
    """
    def __init__(self, output_dir, device_type=None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up PyTorch device (used later for the tensors, not the MSSWD object)
        if device_type:
            self.device = torch.device(device_type)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the MS-SWD metric without .to(self.device)
        self.ms_swd = MSSWD(
            n_levels=3,
            n_descriptors=512,
            descriptor_size=(3, 7, 7),
            n_repeat=12,
            n_proj=128,
            padding_mode='circular',
            combine_levels=True
            # Note: The original author included `n_gpu=2` here. 
            # You can add `n_gpu=1` if you are using a single GPU and experience issues.
        )

    def _prepare_tensors(self, data_list):
        """Converts a list of 3D numpy arrays into PyTorch tensors with a channel dimension."""
        # Expected shape for 3D convs: (N, C, Z, Y, X)
        arr = np.stack(data_list)
        
        # If the data is categorical (1, 4, 8), you might want to normalize it 
        # or convert it to one-hot depending on how the GAN was trained.
        # Assuming single channel continuous/categorical for now:
        if arr.ndim == 4:
            arr = np.expand_dims(arr, axis=1) # Add channel dimension
            
        tensor = torch.tensor(arr, dtype=torch.float32).to(self.device)
        return tensor

    def compute_msswd_mds(self, real_data, gan_data, random_state=42, save_data=True, load_existing=False):
        """Computes pairwise MS-SWD and applies MDS to generate 2D embeddings.
        
        Args:
            real_data (list): List of real sample arrays.
            gan_data (list): List of GAN sample arrays.
            random_state (int): Seed for MDS reproducibility.
            save_data (bool): If True, saves the calculated arrays to a .npz file.
            load_existing (bool): If True, attempts to load a previously saved .npz file 
                from the output directory to bypass the computation.
        """
        save_path = os.path.join(self.output_dir, "msswd_mds_data.npz")
        
        # Check if we can load existing data to skip the 70-minute calculation!
        if load_existing and os.path.exists(save_path):
            print(f"Found existing data! Loading embeddings and distances from {save_path}...")
            loaded_data = np.load(save_path)
            return loaded_data['real_embeddings'], loaded_data['gan_embeddings'], loaded_data['distances']

        print(f"Preparing tensors for {len(real_data)} Real and {len(gan_data)} GAN samples...")
        n_real = len(real_data)
        n_gan = len(gan_data)
        total_samples = n_real + n_gan
        
        real_tensor = self._prepare_tensors(real_data)
        gan_tensor = self._prepare_tensors(gan_data)
        all_samples = torch.cat((real_tensor, gan_tensor), dim=0)
        
        # Initialize an empty distance matrix
        distances = np.zeros((total_samples, total_samples))
        
        # Calculate exactly how many comparisons we are making for the progress bar
        total_comparisons = (total_samples * (total_samples - 1)) // 2
        print(f"Computing {total_comparisons} pairwise Multi-Scale Sliced Wasserstein Distances...")
        
        # Note: This scales O(N^2) and can be very slow for large sample sizes
        with torch.no_grad():
            # Initialize tqdm with the total number of calculations
            with tqdm(total=total_comparisons, desc="MS-SWD Progress", unit="pair") as pbar:
                for j in range(total_samples):
                    for k in range(j + 1, total_samples):
                        # Compute distance between sample j and sample k
                        dist = self.ms_swd(all_samples[j:j+1], all_samples[k:k+1]).item()
                        distances[j, k] = dist
                        distances[k, j] = dist # Matrix is symmetric
                        
                        # Update the progress bar by 1 after each comparison
                        pbar.update(1)
                    
        print("Applying Multidimensional Scaling (MDS)...")
        reducer = MDS(
            n_components=2,
            n_jobs=-1, # Use all CPU cores
            random_state=random_state,
            dissimilarity='precomputed'
        )
        
        embeddings = reducer.fit_transform(distances)
        print(f"MDS stress: {reducer.stress_:.4f}")
        
        real_embeddings = embeddings[:n_real]
        gan_embeddings = embeddings[n_real:]
        
        # Save the computed data to the output directory
        if save_data:
            np.savez(
                save_path, 
                real_embeddings=real_embeddings, 
                gan_embeddings=gan_embeddings, 
                distances=distances
            )
            print(f"Successfully saved embeddings and distance matrix to: {save_path}")
        
        return real_embeddings, gan_embeddings, distances

    def plot_mds_embeddings(self, real_embeddings, gan_embeddings, save_plot=True):
        """Generates a scatter plot of the MDS embeddings."""
        print("\n--- Generating MS-SWD MDS Plot ---")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use styling similar to Figure 2 of the paper
        ax.scatter(
            real_embeddings[:, 0], real_embeddings[:, 1], 
            s=30, c='#d1d1d1', marker='s', lw=0.5, ec='white', label='Real (Test) Samples'
        )
        ax.scatter(
            gan_embeddings[:, 0], gan_embeddings[:, 1], 
            s=30, c='#003f5c', marker='o', lw=0.5, ec='white', label='GAN Samples'
        )
        
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('MS-SWD Multidimensional Scaling', fontsize=14, pad=15)
        
        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='#BFBFBF', alpha=0.5, linestyle='--', linewidth=0.5)
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, "msswd_mds_plot.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved MDS plot to: {plot_path}")
            
        plt.show()

if __name__ == "__main__":
    # 1. Initialize
    validator = PostProcessing(
        output_dir='outputs/20000_training_samples',
        real_path='datasets/training/*.npy',
        gan_path='outputs/realizations/*.npy'
    )

    # 2. Load the Samples into self (Limiting to 10 to save RAM!)
    validator.load_gan_samples(limit=10)
    validatflumy(limit=10)

    # 3. Call functions with explicit save/show logic
    validator.plot_facies_percentages(show_plot=True, save_plot=False)
    validator.connectivity_and_pattern_analysis(target_val=1)
    validator.plot_entropy(show_plot=False, save_plot=True)
    validator.compute_slice_metrics(axis='Z', num_slices=3)
    validator.plot_3d_pyvista(show_plot=True, save_plot=True)