import os
import sys
import glob
import math
import random
import pathlib
import warnings
from pathlib import Path

# Mathematical libraries
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
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import MDS
from voxgan.models.metrics import MSSWD


# Miscellaneous
from tqdm import tqdm
from collections import Counter
from skimage.util import view_as_windows
from collections import Counter, defaultdict

# Adjust path based on your environment
scripts_dir = Path(__file__).resolve().parents[2]
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))


# --- Centralized configuration ---
def get_facies_config():
    """Centralized configuration for facies properties to prevent hardcoding."""
    return {
        'codes': [1, 4, 8],
        'names': {
            1: 'Sand body deposits',
            4: 'Crevasse splay & Levee deposits',
            8: 'Clay deposits'
        },
        'colors': {
            1: '#f1970f',
            4: '#fffc65',
            8: '#33ff00'
        },
        # Maps 0-indexed categorical outputs to physical codes
        'mapping': {0: 1, 1: 4, 2: 8} 
    }


class WellMismatch:
    """Quantifies the degree of mismatch between the 3D realizations and the well data.

    Calculates the Macro F1 score to evaluate how accurately the generated grid 
    captures the well data. Macro F1 is used to account for geological class 
    imbalance, giving equal weight to minority facies (e.g., channels).

    Attributes:
        flumy_name (str): Display name for the Flumy baseline dataset in plots.
        gan_name (str): Display name for the generated dataset in plots.
        cfg (dict): Centralized facies configuration from get_facies_config().
        data_files (list): Sorted list of file paths to the GAN realizations.
        well_data_path (str): Path to the well data file (e.g., CSV or Excel).
        well_coords (list): List of (Z, Y, X) tuples representing well locations.
        well_true_facies (list): List of true physical facies codes at the well coords.
    """
    def __init__(self, flumy_name, gan_name, data_dir, well_data_path):
        """Initializes the WellMismatch evaluator.

        Args:
            flumy_name (str): Display name for the Flumy baseline dataset (e.g., 'Flumy').
            gan_name (str): Display name for the generated dataset (e.g., 'VoxGAN').
            data_dir (str): Glob pattern matching GAN realization files.
            well_data_path (str): Path to the well data Excel file.
        """
        self.flumy_name = flumy_name
        self.gan_name = gan_name
        self.cfg = get_facies_config()
        
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
                df_32 = df.head(32) # Takes first 32 meters (matches Z grid size)
                
                facies_col = 'Facies ' if 'Facies ' in df.columns else 'Facies'
                
                for i, row in df_32.iterrows():
                    z = i
                    y = int(np.round((row['GRID N'] - ORIGIN_N) / SPACING))
                    x = int(np.round((row['GRID E'] - ORIGIN_E) / SPACING))
                    
                    if 0 <= x < 128 and 0 <= y < 128:
                        raw_facies = row[facies_col]
                        
                        if 1 <= raw_facies <= 3: 
                            mapped_facies = 1
                        elif 4 <= raw_facies <= 7: 
                            mapped_facies = 4
                        elif 8 <= raw_facies <= 12: 
                            mapped_facies = 8
                        else:
                            mapped_facies = 1
                            
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
            
        if raw_arr.ndim == 4:
            class_indices = np.argmax(raw_arr, axis=0) 
        elif raw_arr.ndim == 3:
            class_indices = np.round(raw_arr).astype(int)
        else:
            raise ValueError(f"Unexpected array shape {raw_arr.shape} in {file_path}")

        mapping = np.array(self.cfg['codes'])
        class_indices = np.clip(class_indices, 0, len(mapping) - 1)
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
                
                for (z, y, x), true_facies in zip(self.well_coords, self.well_true_facies):
                    if 0 <= z < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= x < grid.shape[2]:
                        y_pred.append(grid[z, y, x])
                        y_true_valid.append(true_facies)
                    
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

    def plot_vpc(self, num_realizations=10, figsize=None, save_plot=False, output_dir='outputs'):
        """Calculates and plots the Vertical Proportion Curve (VPC) comparing Well vs GAN data.

        Args:
            num_realizations (int, optional): Number of GAN realizations to average. Defaults to 10.
            figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (12, 8).
            save_plot (bool, optional): If True, saves the plot to output_dir. Defaults to False.
            output_dir (str, optional): Directory to save the plot. Defaults to 'outputs'.
        """
        print(f"\n--- Generating Vertical Proportion Curves (VPC) ---")
        if not self.well_coords:
            print("No well data loaded. Cannot compute VPC.")
            return

        figsize = figsize or (12, 8)
        facies_codes = self.cfg['codes']
        facies_colors = self.cfg['colors']
        facies_labels = self.cfg['names']

        max_z = 32

        well_vpc = {f: np.zeros(max_z) for f in facies_codes}
        well_counts_per_z = np.zeros(max_z)

        for (z, y, x), f_val in zip(self.well_coords, self.well_true_facies):
            if z < max_z:
                well_vpc[f_val][z] += 1
                well_counts_per_z[z] += 1

        for f in facies_codes:
            safe_counts = np.where(well_counts_per_z == 0, 1, well_counts_per_z)
            well_vpc[f] = well_vpc[f] / safe_counts
            well_vpc[f][well_counts_per_z == 0] = np.nan 

        gan_vpc = {f: np.zeros(max_z) for f in facies_codes}
        files_to_process = self.data_files[:num_realizations]
        
        print(f"Processing {len(files_to_process)} {self.gan_name} realizations for global VPC...")
        for file_path in files_to_process:
            grid = self._load_and_map_realization(file_path)
            z_dim, y_dim, x_dim = grid.shape
            cells_per_slice = y_dim * x_dim
            
            for f in facies_codes:
                proportions = np.sum(grid == f, axis=(1, 2)) / cells_per_slice
                gan_vpc[f] += proportions[:max_z]
                
        for f in facies_codes:
            gan_vpc[f] /= len(files_to_process)

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        z_array = np.arange(max_z)

        for f in facies_codes:
            axes[0].plot(well_vpc[f], z_array, color=facies_colors[f], 
                         label=facies_labels[f], linewidth=2.5, marker='o', markersize=4)
        
        axes[0].set_title(f'{self.flumy_name} VPC (Local)', fontsize=14)
        axes[0].set_ylabel('Depth (Z-index)', fontsize=12)
        axes[0].set_xlabel('Proportion', fontsize=12)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 1)
        axes[0].grid(True, linestyle='--', alpha=0.7)

        for f in facies_codes:
            axes[1].plot(gan_vpc[f], z_array, color=facies_colors[f], 
                         linewidth=2.5)
            axes[1].fill_betweenx(z_array, 0, gan_vpc[f], color=facies_colors[f], alpha=0.1)

        axes[1].set_title(f'{self.gan_name} VPC (Global Mean of {len(files_to_process)} samples)', fontsize=14)
        axes[1].set_xlabel('Proportion', fontsize=12)
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        handles = [plt.Line2D([0], [0], color=facies_colors[f], lw=4) for f in facies_codes]
        fig.legend(handles, [facies_labels[f] for f in facies_codes], 
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=12)

        plt.suptitle(f"Vertical Proportion Curve: {self.flumy_name} vs. {self.gan_name}", fontsize=16)
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"vpc_comparison_{num_realizations}_samples.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved VPC plot to: {plot_path}")

        plt.show()

    def plot_roc_curve(self, save_plot=False, output_dir='outputs'):
        """Calculates and plots a Multi-Class Ensemble ROC Curve against well data."""
        print(f"\n--- Generating Ensemble ROC Curve ---")
        if not self.well_coords:
            print("No well data loaded. Cannot compute ROC.")
            return

        facies_codes = self.cfg['codes']
        y_true_bin = label_binarize(self.well_true_facies, classes=facies_codes)
        n_classes = len(facies_codes)

        num_samples = len(self.data_files)
        well_probs = np.zeros((len(self.well_coords), n_classes))

        print(f"Calculating voxel probabilities across {num_samples} realizations...")
        for file_path in self.data_files:
            grid = self._load_and_map_realization(file_path)
            
            for i, (z, y, x) in enumerate(self.well_coords):
                if 0 <= z < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= x < grid.shape[2]:
                    pred_facies = grid[z, y, x]
                    
                    if pred_facies in facies_codes:
                        class_idx = facies_codes.index(pred_facies)
                        well_probs[i, class_idx] += 1
        
        well_probs /= num_samples

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        facies_names = self.cfg['names']
        facies_colors = self.cfg['colors']

        fig, ax = plt.subplots(figsize=(8, 8))

        for i, code in enumerate(facies_codes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], well_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            ax.plot(fpr[i], tpr[i], color=facies_colors[code], lw=2.5,
                    label=f"{facies_names[code]} (AUC = {roc_auc[i]:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guessing (AUC = 0.5)')
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Ensemble ROC Curve: {self.gan_name} vs. Well Data\n({num_samples} Realizations Evaluated)', fontsize=14, pad=15)
        
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.4, linestyle='--')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"roc_curve_{num_samples}_samples.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved ROC plot to: {plot_path}")

        plt.show()


class PostProcessing:
    """Handles spatial and statistical validation metrics for GAN-generated geological facies.

    This class loads 3D arrays of geological facies (both Flumy samples and GAN-generated),
    computes spatial statistics (MPS, connectivity, normalized entropy, JSD), 
    and generates visual diagnostic plots including 3D renders.

    Attributes:
        flumy_name (str): Display name for the Flumy baseline dataset in plots.
        gan_name (str): Display name for the generated dataset in plots.
        output_dir (str): Directory where generated plots and CSVs will be saved.
        data_files (list): Sorted list of file paths for the Flumy dataset.
        gan_files (list): Sorted list of file paths for the GAN-generated dataset.
        flumy_samples (list): Loaded Flumy data arrays in memory.
        gan_data (list): Loaded GAN data arrays in memory.
        cfg (dict): Centralized facies configuration from get_facies_config().
        facies_mapping (dict): Mapping from categorical indices to specific geological codes.
    """
    
    def __init__(self, flumy_name, gan_name, output_dir, data_path, gan_path):
        """Initializes the class and locates files, but defers loading to save memory.

        Args:
            flumy_name (str): Display name for the Flumy baseline dataset in plots (e.g., 'Flumy Data').
            gan_name (str): Display name for the generated dataset in plots.
            output_dir (str): Directory to save outputs (e.g., 'outputs/metrics').
            data_path (str): Glob pattern matching Flumy data files (e.g., 'datasets/*.npy').
            gan_path (str): Glob pattern matching GAN data files (e.g., 'outputs/*.npy').
        """ 
        self.flumy_name = flumy_name
        self.gan_name = gan_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.script_path = pathlib.Path(__file__).resolve()
        self.cwd = pathlib.Path().resolve()
        
        self.data_files = sorted(glob.glob(str(data_path)))
        self.gan_files = sorted(glob.glob(str(gan_path)))
        
        self.flumy_samples = []
        self.gan_data = []
        
        self.cfg = get_facies_config()
        self.facies_mapping = self.cfg['mapping']

    
    def load_flumy_samples(self, limit=100):
        """Loads Flumy data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading {self.flumy_name} samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.data_files[:limit] if limit else self.data_files
        self.flumy_samples = [self._load_data(f) for f in files_to_load]
        print(f"Loaded {len(self.flumy_samples)} {self.flumy_name} samples into memory.")

    def load_gan_samples(self, limit=100):
        """Loads GAN data into memory.
        
        Args:
            limit (int, optional): Maximum files to load to prevent out-of-memory errors.
                Defaults to 100. Pass None to load all matching files.
        """
        print(f"Loading {self.gan_name} samples (limit: {limit if limit else 'ALL'})...")
        files_to_load = self.gan_files[:limit] if limit else self.gan_files
        self.gan_data = [self._load_gan(f) for f in files_to_load]
        print(f"Loaded {len(self.gan_data)} {self.gan_name} samples into memory.")

    def _load_data(self, file):
        """Loads a Flumy facies array and maps labels to geological codes.

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
            mapping = np.array(self.cfg['codes'])
        else:
            mapping = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
        return mapping[class_indices]

    def _get_global_proportions(self, data_list, target_facies=None):
        """Calculates the global probability of each facies across the entire dataset.

        Args:
            data_list (list): List of 3D numpy arrays containing geological facies.
            target_facies (list, optional): List of facies codes to calculate proportions for.
                Defaults to None, which uses self.cfg['codes'].

        Returns:
            list: Float probabilities of each target facies, summing to 1.0.
        """
        if target_facies is None:
            target_facies = self.cfg['codes']
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
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _create_colormap_and_legend(self):
        """Creates a ListedColormap and legend patches from the centralized facies config.

        Returns:
            tuple: (ListedColormap, list of matplotlib Patch objects for legends).
        """
        color_list = [self.cfg['colors'][c] for c in self.cfg['codes']]
        custom_cmap = ListedColormap(color_list)
        legend_patches = [
            mpatches.Patch(color=self.cfg['colors'][c], label=self.cfg['names'][c]) 
            for c in self.cfg['codes']
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

    def connectivity_and_pattern_analysis(self, facies_list=None, sample_limit=10, recompute_flumy=False):
        """Executes MPS and Connectivity comparisons for ALL facies using loaded data.
        
        Flumy baseline data is computed once and saved to a CSV. On subsequent runs, 
        it loads the baseline from the CSV to save time, only recomputing the GAN data.

        Args:
            facies_list (list, optional): Specific facies to analyze. Defaults to config codes.
            sample_limit (int, optional): Maximum number of samples to process. 
                Defaults to 10. Pass None to process all.
            recompute_flumy (bool, optional): If True, forces a recalculation of the 
                Flumy baseline dataset even if a saved CSV exists. Defaults to False.
                
        Returns:
            tuple: (DataFrame of connectivity stats, Flumy Pattern Counter, GAN Pattern Counter)
        """
        if facies_list is None:
            facies_list = self.cfg['codes']

        limit_str = str(sample_limit) if sample_limit is not None else "ALL"
        
        baseline_csv_path = os.path.join(self.output_dir, f"baseline_connectivity_{limit_str}_samples.csv")
        gan_csv_path = os.path.join(self.output_dir, f"connectivity_comparison_{limit_str}_samples.csv")

        flumy_stats_dict = {}
        flumy_pattern_counter = Counter()
        total_flumy_patterns = "Loaded from CSV (Not Recomputed)"

        if not recompute_flumy and os.path.exists(baseline_csv_path):
            print(f"\n--- Loading pre-computed {self.flumy_name} metrics from: {os.path.basename(baseline_csv_path)} ---")
            df_baseline = pd.read_csv(baseline_csv_path)
            
            for _, row in df_baseline.iterrows():
                flumy_stats_dict[row['Facies']] = {
                    'Flumy_Max_Blob': row['Flumy_Max_Blob'],
                    'Flumy_Median': row['Flumy_Median'],
                    'Flumy_Mean': row['Flumy_Mean']
                }
        else:
            print(f"\n--- Computing {self.flumy_name} metrics for {limit_str} samples ---")
            if getattr(self, 'flumy_samples', None) is None or len(self.flumy_samples) == 0:
                self.load_flumy_samples(limit=sample_limit)
            
            flumy_to_process = self.flumy_samples[:sample_limit] if sample_limit else self.flumy_samples
            all_flumy_blobs = defaultdict(list)
            
            for data in flumy_to_process:
                patterns, counts = self.get_pattern_counts(data)
                for p, c in zip(patterns, counts):
                    flumy_pattern_counter[tuple(p)] += c 
                    
                for f_val in facies_list:
                    all_flumy_blobs[f_val].extend(self.analyze_connectivity(data, f_val))
                    
            total_flumy_patterns = len(flumy_pattern_counter)
            
            baseline_rows = []
            for f_val in facies_list:
                d_blobs = all_flumy_blobs[f_val]
                stats = {
                    'Flumy_Max_Blob': max(d_blobs) if d_blobs else 0,
                    'Flumy_Median': np.median(d_blobs) if d_blobs else 0,
                    'Flumy_Mean': np.mean(d_blobs) if d_blobs else 0
                }
                flumy_stats_dict[f_val] = stats
                
                row = {'Facies': f_val}
                row.update(stats)
                baseline_rows.append(row)
                
            pd.DataFrame(baseline_rows).to_csv(baseline_csv_path, index=False)
            print(f"Saved {self.flumy_name} baseline metrics to: {baseline_csv_path}")

        print(f"\n--- Computing {self.gan_name} metrics for {limit_str} samples ---")
        if getattr(self, 'gan_data', None) is None or len(self.gan_data) == 0:
            self.load_gan_samples(limit=sample_limit)
            
        gan_to_process = self.gan_data[:sample_limit] if sample_limit else self.gan_data
        gan_pattern_counter = Counter()
        all_gan_blobs = defaultdict(list)

        for data in gan_to_process:
            patterns, counts = self.get_pattern_counts(data)
            for p, c in zip(patterns, counts):
                gan_pattern_counter[tuple(p)] += c
                
            for f_val in facies_list:
                all_gan_blobs[f_val].extend(self.analyze_connectivity(data, f_val))

        stats_data = []
        for f_val in facies_list:
            g_blobs = all_gan_blobs[f_val]
            
            d_stats = flumy_stats_dict.get(f_val, {'Flumy_Max_Blob': 0, 'Flumy_Median': 0, 'Flumy_Mean': 0})
            
            stats_data.append({
                'Facies': f_val,
                'Flumy_Max_Blob': d_stats['Flumy_Max_Blob'],
                'GAN_Max_Blob': max(g_blobs) if g_blobs else 0,
                'Flumy_Median': d_stats['Flumy_Median'],
                'GAN_Median': np.median(g_blobs) if g_blobs else 0,
                'Flumy_Mean': d_stats['Flumy_Mean'],
                'GAN_Mean': np.mean(g_blobs) if g_blobs else 0
            })
            
        df_stats = pd.DataFrame(stats_data)
        df_stats['Flumy_Mean'] = df_stats['Flumy_Mean'].round(2)
        df_stats['GAN_Mean'] = df_stats['GAN_Mean'].round(2)
        
        df_stats.to_csv(gan_csv_path, index=False)

        print("\n" + "="*50)
        print(" MULTIPLE POINT STATISTICS (MPS)")
        print("="*50)
        print(f"Total unique patterns ({self.flumy_name}): {total_flumy_patterns}")
        print(f"Total unique patterns ({self.gan_name}): {len(gan_pattern_counter)}")
        
        print("\n" + "="*50)
        print(" MACRO-CONNECTIVITY STATISTICS")
        print("="*50)
        print(df_stats.to_string(index=False))
        print(f"\nSaved full comparison metrics to: {gan_csv_path}\n")

        return df_stats, flumy_pattern_counter, gan_pattern_counter

    def plot_facies_percentages(self, mode='gan', figsize=None, show_plot=True, save_plot=False):
        """Plots volume percentages of specific facies globally across the dataset.

        Args:
            mode (str, optional): Target to evaluate ('gan', 'flumy', or 'both'). Defaults to 'gan'.
            figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (10, 6).
            show_plot (bool, optional): If True, displays the plot interactively. Defaults to True.
            save_plot (bool, optional): If True, saves the plot to the output directory. Defaults to False.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        valid_modes = ['gan', 'flumy', 'both']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        if mode in ['gan', 'both'] and not self.gan_data:
            self.load_gan_samples()
        if mode in ['flumy', 'both'] and not self.flumy_samples:
            self.load_flumy_samples()
            
        print(f"\n--- Generating Facies Distribution (Mode: {mode.upper()}) ---")

        figsize = figsize or (10, 6)
        target_facies = self.cfg['codes']
        labels = [self.cfg['names'][c] for c in self.cfg['codes']]
        colors = [self.cfg['colors'][c] for c in self.cfg['codes']]
        
        def calculate_percentages(data_list):
            probs = self._get_global_proportions(data_list, target_facies)
            return [p * 100 for p in probs]

        fig, ax = plt.subplots(figsize=figsize)

        if mode == 'both':
            flumy_percentages = calculate_percentages(self.flumy_samples)
            gan_percentages = calculate_percentages(self.gan_data)
            
            x = np.arange(len(labels))
            width = 0.35 
            
            bars_flumy = ax.bar(x - width/2, flumy_percentages, width, color=colors, edgecolor='black', alpha=0.5)
            bars_gan = ax.bar(x + width/2, gan_percentages, width, color=colors, edgecolor='black', hatch='//')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            
            for bar in bars_flumy:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, color='#555555')
            for bar in bars_gan:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
            legend_elements = [
                mpatches.Patch(facecolor='gray', alpha=0.5, edgecolor='black', label=f'{self.flumy_name} ({len(self.flumy_samples)} samples)'),
                mpatches.Patch(facecolor='gray', hatch='//', edgecolor='black', label=f'{self.gan_name} ({len(self.gan_data)} samples)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
            ax.set_title(f'Facies Volume Distribution: {self.flumy_name} vs. {self.gan_name}', fontsize=14, pad=15)
            
        else:
            target_data = self.gan_data if mode == 'gan' else self.flumy_samples
            percentages = calculate_percentages(target_data)
            
            bars = ax.bar(labels, percentages, color=colors, edgecolor='black', linewidth=1.2)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
            display_name = self.gan_name if mode == 'gan' else self.flumy_name
            ax.set_title(f'{display_name} Facies Volume Distribution ({len(target_data)} Samples)', fontsize=14, pad=15)

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

    def plot_entropy(self, data_source='gan', axis=None, num_slices=9, figsize=None, plot_title=True, show_plot=True, save_plot=False):
        """Calculates and plots cell-wise normalized spatial entropy across the dataset.

        Args:
            data_source (str, optional): Target to evaluate ('gan' or 'flumy'). Defaults to 'gan'.
            axis (str, optional): Axis to slice ('X', 'Y', 'Z'). If None, processes all. Defaults to None.
            num_slices (int, optional): Slices to plot per axis. Defaults to 9.
            figsize (tuple, optional): Figure size as (width, height) in inches. Passed to subplot helper.
                If None, auto-computed based on num_slices.
            plot_title (bool, optional): If True, displays a suptitle. Defaults to True.
            show_plot (bool, optional): If True, displays the plot interactively. Defaults to True.
            save_plot (bool, optional): If True, saves the plot to the output directory. Defaults to False.

        Raises:
            ValueError: If input arguments do not match expected constraints.
        """
        valid_sources = ['gan', 'flumy']
        if data_source not in valid_sources:
            raise ValueError(f"Invalid data_source '{data_source}'. Choose from {valid_sources}.")
            
        if axis:
            axis = axis.upper()
            if axis not in ['X', 'Y', 'Z']:
                raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
        
        if data_source == 'gan' and not self.gan_data:
            self.load_gan_samples()
        
        # Flumy data is always needed to compute the baseline H_max
        if not self.flumy_samples:
            self.load_flumy_samples()
            
        target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
        num_total = len(target_data)
        axes_to_plot = [axis] if axis else ['Z', 'Y', 'X']

        flumy_global_probs = self._get_global_proportions(self.flumy_samples)
        h_max = self._calculate_h_max(flumy_global_probs) * 1.3
        
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        print(f"\n--- Generating {display_name} Normalized Entropy Matrices ---")
        print(f"{self.flumy_name} Baseline Proportions: {[round(p, 4) for p in flumy_global_probs]}")
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

        if 'Z' in stacks: self._plot_entropy_helper(stacks['Z'], slices_dict['Z'], 'Z', 'X', 'Y', data_source, h_max, figsize, plot_title, show_plot, save_plot)
        if 'Y' in stacks: self._plot_entropy_helper(stacks['Y'], slices_dict['Y'], 'Y', 'X', 'Z', data_source, h_max, figsize, plot_title, show_plot, save_plot)
        if 'X' in stacks: self._plot_entropy_helper(stacks['X'], slices_dict['X'], 'X', 'Y', 'Z', data_source, h_max, figsize, plot_title, show_plot, save_plot)

    def _plot_entropy_helper(self, slices_stack, slice_indices, axis_name, xlabel, ylabel, data_source, h_max, figsize, plot_title, show_plot, save_plot):
        """Internal helper to compute, format, and render normalized entropy charts.

        Args:
            slices_stack (numpy.ndarray): Stacked 2D cross-sections across all realizations.
            slice_indices (list): Integer indices indicating absolute slice depth in the 3D grid.
            axis_name (str): Label of the slicing axis ('X', 'Y', or 'Z').
            xlabel (str): Label for the horizontal axis of the output plot.
            ylabel (str): Label for the vertical axis of the output plot.
            data_source (str): Identifier string ('gan' or 'flumy') for title formatting.
            h_max (float): The theoretical maximum entropy (bits) for normalization.
            figsize (tuple or None): Figure size as (width, height). If None, auto-computed.
            plot_title (bool): Display suptitle if True.
            show_plot (bool): Display plot interactively if True.
            save_plot (bool): Save plot to output directory if True.
        """
        num_realizations, n_slices, dim_y, dim_x = slices_stack.shape
        facies_values = self.cfg['codes']
        norm = mcolors.Normalize(vmin=0, vmax=1.0) 

        if figsize:
            ncols = min(n_slices, 4)
            nrows = math.ceil(n_slices / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        elif n_slices == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        elif n_slices <= 3:
            fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
        else:
            ncols = min(n_slices, 4)
            nrows = math.ceil(n_slices / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        axes_list = np.atleast_1d(axes).flatten()

        for idx, slice_val in enumerate(slice_indices):
            probs = np.zeros((len(facies_values), dim_y, dim_x))
            for i, f_val in enumerate(facies_values):
                probs[i] = np.sum(slices_stack[:, idx, :, :] == f_val, axis=0) / num_realizations
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entropy_map = entropy(probs, base=2, axis=0)
                if h_max > 0:
                    entropy_map = entropy_map / h_max
                
            im = axes_list[idx].imshow(entropy_map, cmap='magma', origin='lower', norm=norm)
            axes_list[idx].set_title(f"{axis_name}-Slice = {slice_val}")
            
            ncols_actual = min(n_slices, 4) if n_slices > 3 else n_slices
            if idx % ncols_actual == 0:
                axes_list[idx].set_ylabel(ylabel)
            nrows_actual = math.ceil(n_slices / ncols_actual)
            if idx >= (nrows_actual - 1) * ncols_actual:
                axes_list[idx].set_xlabel(xlabel)

        for idx in range(n_slices, len(axes_list)):
            axes_list[idx].set_visible(False)

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label('Normalized Entropy (0 to 1)', rotation=270, labelpad=15)
        
        plane = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}[axis_name]
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        if plot_title:
            plt.suptitle(f"{display_name}: {plane} Plane Normalized Cell-Wise Entropy ({num_realizations} Realizations)", fontsize=16)

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
        dz = np.pad(np.diff(data_array, axis=0), ((0, 1), (0, 0), (0, 0)), mode='edge')
        dy = np.pad(np.diff(data_array, axis=1), ((0, 0), (0, 1), (0, 0)), mode='edge')
        dx = np.pad(np.diff(data_array, axis=2), ((0, 0), (0, 0), (0, 1)), mode='edge')

        gradients = np.vstack([dz.flatten(), dy.flatten(), dx.flatten()]).T

        _, counts = np.unique(gradients, axis=0, return_counts=True)
        
        probabilities = counts / counts.sum()

        # The paper halves the 2D entropy to account for Papoulis' sampling redundancy, 
        # but for comparative metrics between datasets, the raw joint entropy is an excellent structural indicator.
        spatial_entropy = entropy(probabilities, base=2)
        
        # We divide by 2 following the parsimony principle outlined for Delentropy 
        # to halve the vector data rate using generalized sampling expansion.
        delentropy = spatial_entropy / 2.0 

        return delentropy

    def compare_structural_delentropy(self):
        """Evaluates and compares the spatial Delentropy of Flumy vs GAN datasets."""
        if not self.gan_data: self.load_gan_samples()
        if not self.flumy_samples: self.load_flumy_samples()

        print(f"\n--- Computing 3D Spatial Delentropy (Structural Complexity) ---")
        
        flumy_entropies = [self.compute_spatial_delentropy(arr) for arr in self.flumy_samples]
        gan_entropies = [self.compute_spatial_delentropy(arr) for arr in self.gan_data]

        flumy_mean = np.mean(flumy_entropies)
        gan_mean = np.mean(gan_entropies)
        
        print(f"{self.flumy_name} Mean Delentropy: {flumy_mean:.4f} bits")
        print(f"{self.gan_name} Mean Delentropy:  {gan_mean:.4f} bits")
        print(f"Difference ({self.gan_name} - {self.flumy_name}):   {gan_mean - flumy_mean:.4f} bits")
        
        if abs(gan_mean - flumy_mean) < 0.1:
            print(f"-> The {self.gan_name} exhibits excellent spatial structural fidelity compared to {self.flumy_name}.")
        else:
            print("-> A notable difference in spatial entropy suggests structural deviations (e.g., over-smoothing or excessive noise).")
            
        return {'flumy_delentropy_mean': flumy_mean, 'gan_delentropy_mean': gan_mean}

    def plot_local_delentropy_map(self, data_source='gan', slice_idx=None, window_size=5, figsize=None):
        """Generates and plots a 2D spatial map of local Delentropy using a sliding window.
        
        Args:
            data_source (str): 'gan' or 'flumy'.
            slice_idx (int, optional): The Z-slice index to plot. If None, picks the middle slice.
            window_size (int, optional): The size of the sliding window (must be odd, e.g., 5, 7, 9).
            figsize (tuple, optional): Figure size as (width, height) in inches. Defaults to (14, 6).
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number (e.g., 3, 5, 7).")

        if data_source == 'gan' and not self.gan_data: self.load_gan_samples()
        if data_source == 'flumy' and not self.flumy_samples: self.load_flumy_samples()
        
        target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
        volume = target_data[random.randint(0, len(target_data) - 1)]
        
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
            
        slice_2d = volume[slice_idx, :, :]
        ny, nx = slice_2d.shape
        
        pad_w = window_size // 2
        padded_slice = np.pad(slice_2d, pad_w, mode='edge')
        
        windows = view_as_windows(padded_slice, (window_size, window_size))
        
        local_entropy_map = np.zeros((ny, nx))
        
        print(f"Calculating local Delentropy map (Window: {window_size}x{window_size})...")
        
        for y in range(ny):
            for x in range(nx):
                patch = windows[y, x]
                
                dy = np.diff(patch, axis=0)[:, :-1].flatten()
                dx = np.diff(patch, axis=1)[:-1, :].flatten()
                
                gradients = np.vstack([dy, dx]).T
                
                _, counts = np.unique(gradients, axis=0, return_counts=True)
                probs = counts / counts.sum()
                
                # Halve the entropy for parsimony (as defined in the Delentropy literature)
                local_entropy_map[y, x] = entropy(probs, base=2) / 2.0

        figsize = figsize or (14, 6)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        custom_cmap, legend_patches = self._create_colormap_and_legend()
        display_slice = np.zeros_like(slice_2d)
        display_slice[slice_2d == 4] = 1
        display_slice[slice_2d == 8] = 2
        im0 = axes[0].imshow(display_slice, cmap=custom_cmap, origin='lower')
        axes[0].legend(handles=legend_patches, loc='upper right')
            
        axes[0].set_title(f"Original Facies (Z-Slice {slice_idx})")
        
        im1 = axes[1].imshow(local_entropy_map, cmap='magma', origin='lower')
        axes[1].set_title(f"Local Delentropy Map ({window_size}x{window_size} Window)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Local Spatial Entropy (bits)")
        
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        plt.suptitle(f"{display_name} - Spatial Structural Mapping", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def compute_slice_metrics(self, axis='Z', plot=False):
        """Computes slice-wise aggregate scalar metrics comparing GAN to Flumy distributions.
        
        Evaluates Mean Normalized Entropy and Jensen-Shannon Divergence (JSD) 
        across EVERY slice in the specified axis.

        Args:
            axis (str, optional): Target axis to slice ('X', 'Y', 'Z'). Defaults to 'Z'.
            plot (bool, optional): If True, displays a distribution plot of the metrics.

        Returns:
            tuple: 
                - pandas.DataFrame: Contains Slice Index, Flumy/GAN Entropy, and JSD for ALL slices.
                - dict: Mean and Standard Deviation for Entropy and JSD across the axis.
        """
        if not self.gan_data: self.load_gan_samples()
        if not self.flumy_samples: self.load_flumy_samples()
            
        print(f"\n--- Computing Scalar Metrics for ALL {axis}-Axis Slices ---")
        
        flumy_global_probs = self._get_global_proportions(self.flumy_samples)
        h_max = self._calculate_h_max(flumy_global_probs)
        
        facies_values = self.cfg['codes']
        num_flumy_samples = len(self.flumy_samples)
        num_gan_samples = len(self.gan_data)
        
        nz, ny, nx = self.flumy_samples[0].shape
        dims = {'Z': nz, 'Y': ny, 'X': nx}
        
        slice_indices = range(dims[axis])
        results = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            with tqdm(total=len(slice_indices), desc=f"JS-entropy 2D ({axis}-axis)") as pbar:
                for slice_idx in slice_indices:
                    if axis == 'Z':
                        flumy_slices = np.array([d[slice_idx, :, :] for d in self.flumy_samples])
                        gan_slices = np.array([d[slice_idx, :, :] for d in self.gan_data])
                    elif axis == 'Y':
                        flumy_slices = np.array([d[:, slice_idx, :] for d in self.flumy_samples])
                        gan_slices = np.array([d[:, slice_idx, :] for d in self.gan_data])
                    else: 
                        flumy_slices = np.array([d[:, :, slice_idx] for d in self.flumy_samples])
                        gan_slices = np.array([d[:, :, slice_idx] for d in self.gan_data])
                        
                    dim_y, dim_x = flumy_slices.shape[1], flumy_slices.shape[2]
                    
                    p_flumy = np.zeros((len(facies_values), dim_y, dim_x))
                    p_gan = np.zeros((len(facies_values), dim_y, dim_x))
                    
                    for i, f_val in enumerate(facies_values):
                        p_flumy[i] = np.sum(flumy_slices == f_val, axis=0) / num_flumy_samples
                        p_gan[i] = np.sum(gan_slices == f_val, axis=0) / num_gan_samples
                        
                    entropy_flumy = np.nanmean(entropy(p_flumy, base=2, axis=0))
                    if h_max > 0: entropy_flumy /= h_max
                    
                    entropy_gan = np.nanmean(entropy(p_gan, base=2, axis=0))
                    if h_max > 0: entropy_gan /= h_max
                    
                    js_distance = jensenshannon(p_flumy, p_gan, base=2, axis=0)
                    mean_jsd = np.nanmean(js_distance ** 2)
                    
                    results.append({
                        'Axis': axis,
                        'Slice_Index': slice_idx,
                        'Flumy_Norm_Entropy': entropy_flumy,
                        'GAN_Norm_Entropy': entropy_gan,
                        'JSD_Flumy_vs_GAN': mean_jsd
                    })
                    pbar.update(1)
                
        df_results = pd.DataFrame(results)
        
        summary_stats = {
            'Flumy_Entropy_Mean': df_results['Flumy_Norm_Entropy'].mean(),
            'Flumy_Entropy_Std': df_results['Flumy_Norm_Entropy'].std(),
            'GAN_Entropy_Mean': df_results['GAN_Norm_Entropy'].mean(),
            'GAN_Entropy_Std': df_results['GAN_Norm_Entropy'].std(),
            'JSD_Mean': df_results['JSD_Flumy_vs_GAN'].mean(),
            'JSD_Std': df_results['JSD_Flumy_vs_GAN'].std(),
        }

        print(f"\n--- Summary Statistics ({axis}-Axis) ---")
        print(f"{self.flumy_name} Entropy: {summary_stats['Flumy_Entropy_Mean']:.4f} ± {summary_stats['Flumy_Entropy_Std']:.4f}")
        print(f"{self.gan_name} Entropy:  {summary_stats['GAN_Entropy_Mean']:.4f} ± {summary_stats['GAN_Entropy_Std']:.4f}")
        print(f"JSD (Spatial): {summary_stats['JSD_Mean']:.4f} ± {summary_stats['JSD_Std']:.4f}\n")
        
        csv_path = os.path.join(self.output_dir, f"slice_metrics_normalized_{axis}.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Saved full slice metrics to: {csv_path}")

        if plot:
            self._plot_metric_distributions(df_results, axis)
            
        return df_results, summary_stats
    
    def plot_2d_slices(self, data_source='gan', num_samples=1, num_slices=1, axis=None, 
                       slice_range=None, slice_indices=None, figsize=None, 
                       show_plot=True, save_plot=False):
        """Visualizes 2D cross-sections (Horizontal and Vertical) of 3D realizations.
        
        Args:
            data_source (str): Target dataset to evaluate ('gan' or 'flumy').
            num_samples (int): Number of realizations to plot.
            num_slices (int): Number of slices to extract per realization. Ignored when
                slice_indices is provided.
            axis (str, optional): Specific axis to slice ('X', 'Y', 'Z'). If None, plots all 3.
            slice_range (tuple, optional): Restricts slicing to a sub-range of the axis as
                (start, end). When provided, num_slices evenly-spaced positions are sampled
                within this range. E.g., slice_range=(80, 120) with num_slices=5.
            slice_indices (list, optional): Explicit list of integer slice positions. Overrides
                both num_slices and slice_range when provided.
            figsize (tuple, optional): Figure size as (width, height) in inches. If None,
                auto-computed based on layout.
            show_plot (bool): If True, displays the plot interactively.
            save_plot (bool): If True, saves the plot to the output directory.
        """
        if axis and axis.upper() not in ['X', 'Y', 'Z']:
            raise ValueError("axis must be 'X', 'Y', 'Z', or None.")
            
        axis = axis.upper() if axis else None

        if data_source == 'gan' and not self.gan_data: 
            self.load_gan_samples(limit=num_samples)
        if data_source == 'flumy' and not self.flumy_samples: 
            self.load_flumy_samples(limit=num_samples)
            
        target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
        files_list = self.gan_files if data_source == 'gan' else self.data_files
        
        plot_limit = min(num_samples, len(target_data))
        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name

        colors = [self.cfg['colors'][c] for c in self.cfg['codes']]
        cmap = ListedColormap(colors)
        labels = [self.cfg['names'][c] for c in self.cfg['codes']]
        legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

        for idx in range(plot_limit):
            data_3d = target_data[idx]
            
            display_data = np.zeros_like(data_3d)
            for map_idx, code in enumerate(self.cfg['codes']):
                display_data[data_3d == code] = map_idx

            depth, height, width = display_data.shape

            def _resolve_slice_indices(dim_size):
                """Resolves final slice positions from slice_indices, slice_range, or linspace."""
                if slice_indices is not None:
                    return np.array(slice_indices, dtype=int)
                elif slice_range is not None:
                    return np.linspace(slice_range[0], slice_range[1], num_slices, dtype=int)
                else:
                    return np.linspace(0, dim_size - 1, num_slices, dtype=int)

            z_idx = _resolve_slice_indices(depth)
            y_idx = _resolve_slice_indices(height)
            x_idx = _resolve_slice_indices(width)
            
            effective_num_slices = len(slice_indices) if slice_indices is not None else num_slices
            
            filename = os.path.basename(files_list[idx]) if idx < len(files_list) else f"Sample_{idx}"

            if axis is None:
                effective_figsize = figsize or (18, 5 * effective_num_slices)
                fig, axes = plt.subplots(effective_num_slices, 3, figsize=effective_figsize)
                
                if effective_num_slices == 1:
                    axes = np.expand_dims(axes, axis=0) 
                
                for i in range(effective_num_slices):
                    axes[i, 0].imshow(display_data[z_idx[i], :, :], cmap=cmap, origin='lower', vmin=0, vmax=len(self.cfg['codes'])-1)
                    axes[i, 0].set_title(f"Horizontal Slice (Z={z_idx[i]})")
                    axes[i, 0].set_xlabel("X (Width)")
                    axes[i, 0].set_ylabel("Y (Height)")

                    axes[i, 1].imshow(display_data[:, y_idx[i], :], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=len(self.cfg['codes'])-1) 
                    axes[i, 1].set_title(f"Vertical Section (Y={y_idx[i]})")
                    axes[i, 1].set_xlabel("X (Width)")
                    axes[i, 1].set_ylabel("Z (Depth)")

                    axes[i, 2].imshow(display_data[:, :, x_idx[i]], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=len(self.cfg['codes'])-1)
                    axes[i, 2].set_title(f"Vertical Section (X={x_idx[i]})")
                    axes[i, 2].set_xlabel("Y (Height)")
                    axes[i, 2].set_ylabel("Z (Depth)")

            else:
                ncols = min(effective_num_slices, 4)
                nrows = math.ceil(effective_num_slices / ncols)
                effective_figsize = figsize or (5 * ncols, 5 * nrows)
                
                fig, axes = plt.subplots(nrows, ncols, figsize=effective_figsize)
                axes_list = np.atleast_1d(axes).flatten()

                idx_map = {'Z': z_idx, 'Y': y_idx, 'X': x_idx}
                current_indices = idx_map[axis]

                for i in range(effective_num_slices):
                    ax = axes_list[i]
                    if axis == 'Z':
                        ax.imshow(display_data[current_indices[i], :, :], cmap=cmap, origin='lower', vmin=0, vmax=len(self.cfg['codes'])-1)
                        ax.set_title(f"Horizontal Slice (Z={current_indices[i]})")
                        ax.set_xlabel("X (Width)")
                        ax.set_ylabel("Y (Height)")
                    elif axis == 'Y':
                        ax.imshow(display_data[:, current_indices[i], :], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=len(self.cfg['codes'])-1)
                        ax.set_title(f"Vertical Section (Y={current_indices[i]})")
                        ax.set_xlabel("X (Width)")
                        ax.set_ylabel("Z (Depth)")
                    elif axis == 'X':
                        ax.imshow(display_data[:, :, current_indices[i]], cmap=cmap, origin='lower', aspect=1, vmin=0, vmax=len(self.cfg['codes'])-1)
                        ax.set_title(f"Vertical Section (X={current_indices[i]})")
                        ax.set_xlabel("Y (Height)")
                        ax.set_ylabel("Z (Depth)")

                for i in range(effective_num_slices, len(axes_list)):
                    axes_list[i].set_visible(False)

            fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=12)
            plt.suptitle(f"{display_name} | Realization: {filename} | Shape: {data_3d.shape}", fontsize=16, fontweight='bold', y=1.02)
            
            if save_plot:
                ax_label = axis if axis else "All_Axes"
                plot_path = os.path.join(self.output_dir, f"2d_slices_{data_source}_{ax_label}_{effective_num_slices}slices_{filename.replace('.npy', '.png')}")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                print(f"Saved 2D slices to: {plot_path}")
                
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

    def plot_3d_pyvista(self, data_source='gan', mode='separate', target_filename=None, target_facies=None, 
                        figsize=None, show_legend=True, show_plot=True, save_plot=False):
        """Renders an interactive or static 3D volumetric plot of a geological sample.

        Args:
            data_source (str, optional): Target dataset to sample ('gan' or 'flumy'). Defaults to 'gan'.
            mode (str, optional): Visual layout mode ('separate' for subplots, or 'combined' for 
                a single overlaid volume). Defaults to 'separate'.
            target_filename (str, optional): Specific filename to load (e.g., 'realization_01.npy'). 
                If None, a random sample is chosen from loaded data.
            target_facies (int or list, optional): Specific facies code(s) to isolate (e.g., 1, 4, or 8). 
                If None, plots all standard facies.
            figsize (tuple, optional): Window size as (width, height) in pixels for the PyVista renderer.
                If None, uses default (800 * num_subplots, 800) for separate or (800, 800) for combined.
            show_legend (bool, optional): Toggles legend in combined mode. Defaults to True.
            show_plot (bool, optional): If True, opens an interactive PyVista window. Defaults to True.
            save_plot (bool, optional): If True, saves an off-screen screenshot. Defaults to False.

        Raises:
            ValueError: If input arguments do not match expected constraints.
        """
        valid_sources = ['gan', 'flumy']
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

        display_name = self.flumy_name if data_source == 'flumy' else self.gan_name
        print(f"\n--- Generating 3D PyVista Plot ({display_name} | Mode: {mode.upper()}) ---")

        facies_colors = self.cfg['colors']
        facies_titles = self.cfg['names']

        plot_identifier = ""
        if target_filename:
            file_list = self.gan_files if data_source == 'gan' else self.data_files
            matched_file = next((f for f in file_list if target_filename in f), None)
            
            if matched_file:
                print(f"Loading specific file: {matched_file}")
                data_3d = self._load_gan(matched_file) if data_source == 'gan' else self._load_data(matched_file)
                plot_identifier = target_filename.split('.')[0]
            else:
                print(f"Warning: '{target_filename}' not found. Falling back to random sample.")
                target_filename = None

        if not target_filename:
            if data_source == 'gan' and not self.gan_data:
                self.load_gan_samples()
            elif data_source == 'flumy' and not self.flumy_samples:
                self.load_flumy_samples() 
                
            target_data = self.gan_data if data_source == 'gan' else self.flumy_samples
            random_idx = random.randint(0, len(target_data) - 1)
            data_3d = target_data[random_idx]
            plot_identifier = f"sample_{random_idx}"

        nz, ny, nx = data_3d.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.cell_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

        if target_facies is None:
            facies_to_plot = self.cfg['codes']
        elif isinstance(target_facies, int):
            facies_to_plot = [target_facies]
        else:
            facies_to_plot = target_facies
            
        if len(facies_to_plot) == 1 and mode == 'separate':
            mode = 'combined'

        base_resolution = 800
        scale_factor = 2
        
        if mode == 'separate':
            num_subplots = len(facies_to_plot)
            default_window_size = (base_resolution * num_subplots, base_resolution)
        else:
            num_subplots = 1
            default_window_size = (base_resolution, base_resolution)

        window_size = figsize if figsize else default_window_size

        if mode == 'separate':
            plotter = pv.Plotter(
                shape=(1, num_subplots), 
                image_scale=scale_factor, 
                off_screen=save_plot and not show_plot, 
                window_size=window_size
            )
        else:
            plotter = pv.Plotter(
                shape=(1, 1), 
                image_scale=scale_factor, 
                off_screen=save_plot and not show_plot, 
                window_size=window_size
            )
            
        plotter.enable_anti_aliasing('msaa') 
        plotter.add_title(title=display_name, font_size=12, color='black')
            
        for i, f_val in enumerate(facies_to_plot):
            if f_val not in facies_colors:
                continue
                
            if mode == 'separate':
                plotter.subplot(0, i)
                plotter.add_text(facies_titles[f_val], font_size=150, color='black', shadow=True) 
                
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
    
    Provides dataset-agnostic computation of pairwise MS-SWD distance matrices
    and multi-dataset MDS visualization. Each dataset's distance matrix and raw
    samples are cached to disk, enabling efficient multi-run comparisons without 
    reloading original data.

    Attributes:
        output_dir (str): Base directory where outputs are saved.
        device (torch.device): PyTorch computation device.
        ms_swd (MSSWD): The Multi-Scale Sliced Wasserstein Distance metric.
    """
    DISTANCE_FILENAME = "msswd_distances.npz"

    def __init__(self, output_dir, device_type=None):
        """Initializes the DistributionEvaluator.

        Args:
            output_dir (str): Base directory to save outputs. Individual dataset
                embeddings are saved to their own subdirectories.
            device_type (str, optional): PyTorch device string ('cuda:0', 'cpu'). 
                If None, auto-detects GPU availability.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if device_type:
            self.device = torch.device(device_type)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.ms_swd = MSSWD(
            n_levels=3,
            n_descriptors=512,
            descriptor_size=(3, 7, 7),
            n_repeat=12,
            n_proj=128,
            padding_mode='circular',
            combine_levels=True
        )

    def _prepare_tensors(self, data_list):
        """Converts a list of 3D numpy arrays into PyTorch tensors with a channel dimension.

        Args:
            data_list (list): List of 3D numpy arrays of geological facies.

        Returns:
            torch.Tensor: 5D tensor of shape (N, 1, Z, Y, X) on the configured device.
        """
        arr = np.stack(data_list)
        
        if arr.ndim == 4:
            arr = np.expand_dims(arr, axis=1)
            
        tensor = torch.tensor(arr, dtype=torch.float32).to(self.device)
        return tensor

    def compute_embedding(self, samples, label, save_dir=None):
        """Computes pairwise MS-SWD distances for a single dataset and caches results.

        Calculates the full pairwise distance matrix between all samples using
        MS-SWD and saves it along with the raw sample arrays to a .npz file
        for later use by plot_multi_mds().

        Args:
            samples (list): List of 3D numpy arrays (geological volumes) for this dataset.
            label (str): Human-readable label for this dataset (e.g., 'Flumy', 'VoxGAN_run1').
                Used in progress bar and print statements.
            save_dir (str, optional): Directory to save the distance file. If None,
                uses self.output_dir. The file is always named 'msswd_distances.npz'.

        Returns:
            numpy.ndarray: Symmetric pairwise distance matrix of shape (N, N).
        """
        target_dir = save_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, self.DISTANCE_FILENAME)

        n_samples = len(samples)
        print(f"\nComputing MS-SWD distance matrix for '{label}' ({n_samples} samples)...")
        
        tensor = self._prepare_tensors(samples)
        distances = np.zeros((n_samples, n_samples))
        
        total_comparisons = (n_samples * (n_samples - 1)) // 2
        
        with torch.no_grad():
            with tqdm(total=total_comparisons, desc=f"MS-SWD [{label}]", unit="pair") as pbar:
                for j in range(n_samples):
                    for k in range(j + 1, n_samples):
                        dist = self.ms_swd(tensor[j:j+1], tensor[k:k+1]).item()
                        distances[j, k] = dist
                        distances[k, j] = dist
                        pbar.update(1)

        samples_array = np.stack(samples)
        np.savez(
            save_path,
            distances=distances,
            samples=samples_array
        )
        print(f"Saved distance matrix ({n_samples}x{n_samples}) and samples to: {save_path}")
        
        return distances

    def plot_multi_mds(self, datasets, random_state=42, figsize=None, 
                       save_plot=True, show_plot=True):
        """Generates a combined MDS scatter plot from multiple dataset distance caches.

        Loads cached distance matrices and raw samples from each dataset directory,
        computes cross-dataset pairwise MS-SWD distances, assembles a combined 
        distance matrix, and applies MDS to produce a unified 2D embedding.

        Args:
            datasets (dict): Mapping of dataset labels to their output directories.
                Each directory must contain 'msswd_distances.npz' (created by 
                compute_embedding). Example:
                    {
                        'Flumy': 'outputs/flumy/',
                        'VoxGAN_run1': 'outputs/run1/',
                        'VoxGAN_run2': 'outputs/run2/'
                    }
            random_state (int, optional): Seed for MDS reproducibility. Defaults to 42.
            figsize (tuple, optional): Figure size as (width, height) in inches. 
                Defaults to (8, 8).
            save_plot (bool, optional): If True, saves the plot to self.output_dir. 
                Defaults to True.
            show_plot (bool, optional): If True, displays the plot interactively. 
                Defaults to True.

        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes) for further customization.

        Raises:
            FileNotFoundError: If any dataset directory is missing its distance file.
                Run compute_embedding() for the missing dataset first.
        """
        print("\n--- Loading cached distance data ---")
        
        all_labels = list(datasets.keys())
        cached_data = {}
        
        for label, data_dir in datasets.items():
            dist_path = os.path.join(data_dir, self.DISTANCE_FILENAME)
            if not os.path.exists(dist_path):
                raise FileNotFoundError(
                    f"Distance file not found at '{dist_path}'. "
                    f"Run compute_embedding() for '{label}' first."
                )
            loaded = np.load(dist_path)
            cached_data[label] = {
                'distances': loaded['distances'],
                'samples': loaded['samples']
            }
            print(f"  Loaded '{label}': {loaded['distances'].shape[0]} samples")

        dataset_sizes = [cached_data[label]['distances'].shape[0] for label in all_labels]
        total_samples = sum(dataset_sizes)
        
        combined_distances = np.zeros((total_samples, total_samples))
        
        # Fill in within-dataset blocks from cached matrices
        offset = 0
        offsets = {}
        for label in all_labels:
            n = cached_data[label]['distances'].shape[0]
            offsets[label] = offset
            combined_distances[offset:offset+n, offset:offset+n] = cached_data[label]['distances']
            offset += n

        # Compute cross-dataset distances
        print("\nComputing cross-dataset MS-SWD distances...")
        for i, label_a in enumerate(all_labels):
            for label_b in all_labels[i+1:]:
                samples_a = cached_data[label_a]['samples']
                samples_b = cached_data[label_b]['samples']
                
                tensor_a = self._prepare_tensors(list(samples_a))
                tensor_b = self._prepare_tensors(list(samples_b))
                
                n_a = len(samples_a)
                n_b = len(samples_b)
                off_a = offsets[label_a]
                off_b = offsets[label_b]
                
                total_cross = n_a * n_b
                with torch.no_grad():
                    with tqdm(total=total_cross, desc=f"Cross [{label_a} × {label_b}]", unit="pair") as pbar:
                        for j in range(n_a):
                            for k in range(n_b):
                                dist = self.ms_swd(tensor_a[j:j+1], tensor_b[k:k+1]).item()
                                combined_distances[off_a + j, off_b + k] = dist
                                combined_distances[off_b + k, off_a + j] = dist
                                pbar.update(1)

        print("\nApplying Multidimensional Scaling (MDS) on combined distance matrix...")
        reducer = MDS(
            n_components=2,
            n_jobs=-1,
            random_state=random_state,
            dissimilarity='precomputed'
        )
        
        embeddings = reducer.fit_transform(combined_distances)
        print(f"MDS stress: {reducer.stress_:.4f}")

        # Plot
        figsize = figsize or (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        color_cycle = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']
        
        offset = 0
        for idx, label in enumerate(all_labels):
            n = cached_data[label]['distances'].shape[0]
            color = color_cycle[idx % len(color_cycle)]
            marker = markers[idx % len(markers)]
            
            ax.scatter(
                embeddings[offset:offset+n, 0],
                embeddings[offset:offset+n, 1],
                s=30, c=[color], marker=marker, lw=0.5, ec='white',
                label=f'{label} ({n} samples)'
            )
            offset += n
        
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('MS-SWD Multidimensional Scaling', fontsize=14, pad=15)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='#BFBFBF', alpha=0.5, linestyle='--', linewidth=0.5)
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=min(len(all_labels), 4), frameon=False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, "msswd_mds_multi_plot.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved multi-MDS plot to: {plot_path}")
            
        if show_plot:
            plt.show()
        
        return fig, ax