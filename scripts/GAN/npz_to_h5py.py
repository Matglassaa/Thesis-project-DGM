"""
This script converts a directory of .npz files containing 3D facies data into a single .h5 file for efficient storage and access.
Each .npz file is expected to contain a single array named 'facies' with shape (Z, Y, X) and integer values representing facies classes.
The resulting .h5 file will have a dataset named 'facies' with shape (N, Z, Y, X), where N is the number of .npz files.
Usage:
    python npz_to_h5py.py --data_dir /path/to/npz_files --output_file /path/to/output/dataset.h5
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Convert directory of .npz files to a single .h5 file")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory with .npz files')
    parser.add_argument('--output_file', type=str, required=True, help='Path and name for the output .h5 file (e.g., dataset.h5)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    npz_files = sorted(glob(os.path.join(args.data_dir, '*.npz')))
    num_files = len(npz_files)
    
    if num_files == 0:
        print(f"Error: No .npz files found in {args.data_dir}")
        return

    print(f"Found {num_files} .npz files. Starting conversion...")

    # Load the first file to get the grid dimensions (Z, Y, X)
    first_file = np.load(npz_files[0])['facies']
    grid_shape = first_file.shape
    dtype = first_file.dtype # Likely uint8 or int64

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
        for i, file_path in enumerate(npz_files):
            dataset[i] = np.load(file_path)['facies']
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{num_files} files...")

    print(f"Success! HDF5 dataset saved to: {args.output_file}")

if __name__ == '__main__':
    main()