"""
This script converts a directory of .npz/.npy files containing 3D facies data into a single .h5 file for efficient storage and access.
Each file is expected to contain a single array named 'facies' with shape (Z, Y, X) and integer values representing facies classes.
The resulting .h5 file will have a dataset named 'facies' with shape (N, Z, Y, X), where N is the number of .npz files.

Example usage on a Linux cluster:
    nohup python -u preprocessing.py --data_dir /path/to/npz_files --output_file /path/to/output/dataset.h5 > preprocessing.log 2>&1 &
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob
import sys

from training_dataset_summary import plot_facies_distribution, plot_entropy
from custom_plots import apply_custom_plotting_flavor, FaciesColorMap

def parse_args():
    parser = argparse.ArgumentParser(description="Convert directory of .npz/.npy files to a single .h5 file")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory with .npz/.npy files')
    parser.add_argument('--output_file', type=str, required=True, help='Path and name for the output .h5 file (e.g., dataset.h5)')
    parser.add_argument('--file_type', type=str, choices=['npz', 'npy'], default='npy', help='Type of input files (default: npz)')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save generated plots (default: same as output_file)')
    parser.add_argument('--num_plot_samples', type=int, default=10, help='Number of realizations to sample for the entropy plot')
    
    return parser.parse_args()

def load_array(filepath, file_type):
    """Helper function to correctly load either .npz or .npy files."""
    if file_type == 'npz':
        return np.load(filepath)['facies']
    else:
        return np.load(filepath) # .npy files return the array directly

def main():
    args = parse_args()
    
    # Grab files based on the chosen file type
    search_pattern = f"*.{args.file_type}"
    files = sorted(glob(os.path.join(args.data_dir, search_pattern)))
    num_files = len(files)
    
    if num_files == 0:
        print(f"Error: No .{args.file_type} files found in {args.data_dir}")
        return

    print(f"Found {num_files} .{args.file_type} files. Starting conversion...")

    # Load the first file to get the grid dimensions (Z, Y, X)
    first_file = load_array(files[0], args.file_type)
    grid_shape = first_file.shape
    dtype = first_file.dtype

    # Create the HDF5 file
    with h5py.File(args.output_file, 'w') as h5f:
        dataset = h5f.create_dataset(
            'facies', 
            shape=(num_files, *grid_shape), 
            dtype=dtype,
            chunks=(1, *grid_shape), # 1 sample per chunk
            compression="lzf" # Light compression to save space without heavy CPU tax
        )

        # Populate the dataset
        for i, file_path in enumerate(files):
            dataset[i] = load_array(file_path, args.file_type)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{num_files} files...")

    print(f"Success! HDF5 dataset saved to: {args.output_file}")

    # check dataset contents
    print("-" * 30)
    with h5py.File(args.output_file, 'r') as f:
        for key in f.keys():
            ds = f[key]
            print(f"Dataset Name : {key}")
            print(f"Shape        : {ds.shape}")
            print(f"Data Type    : {ds.dtype}")
            print(f"Chunk Size   : {ds.chunks}")
            print(f"Compression  : {ds.compression}")
            print("-" * 30)
            
    print("Starting plot generation...")
    apply_custom_plotting_flavor()
    
    plot_dir = args.plot_dir if args.plot_dir else os.path.dirname(args.output_file)
    if not plot_dir:
        plot_dir = "."
    os.makedirs(plot_dir, exist_ok=True)
    
    # Extact the filenames (basenames) for the plotting functions to use
    file_basenames = [os.path.basename(f) for f in files]
    
    plot_facies_distribution(args.data_dir, plot_dir, file_basenames)
    plot_entropy(args.data_dir, plot_dir, file_basenames, args.num_plot_samples)

if __name__ == '__main__':
    main()