"""
Flumy runner script utilizing the native Python API for precise Net-to-Gross control.
Runs an 8-sample test batch in parallel, extracts the MIDDLE 32 meters of a 42m simulation, 
and logs execution times.
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from flumy import *
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description="Convert directory of .npz/.npy files to a single .h5 file")
    parser.add_argument('--num_files', type=int, default=1000, help='Number of Flumy samples to generate (default: 20000)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers (default: 8)')
    parser.add_argument('--ntg', type=float, default=0.67, help='Net-to-Gross ratio for the Flumy simulations (default: 0.67)')
    parser.add_argument('--max_ch_depth', type=int, default=5, help='Maximum channel depth in meters (default: 6)')
    parser.add_argument('--isbx', type=int, default=80, help='Number of grid blocks in the X direction (default: 100)')
    
    return parser.parse_args()

def os_check(num_samples, ntg, max_ch_depth, isbx):
    """
    Checks the operating system and sets the appropriate base data directory.
    Returns:
        str: Base directory for data storage based on the operating system.
    """
    if os.name == 'nt':  # Windows
        BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'training',f'setting_1_nexus_{num_samples}_samples_ntg_{int(ntg*100)}_chdepth_{max_ch_depth}_isbx_{isbx}')
    else:  # Linux
        home_dir = os.path.expanduser("~") 
        BASE_PATH = os.path.join(home_dir, 'data', 'datasets', f'training_dataset_nexus_{num_samples}_samples_ntg_{ntg}_chdepth_{max_ch_depth}_isbx_{isbx}')

    return BASE_PATH

def save_config(config, num_samples, output_dir):
    """
    """
    output_dir = os.path.join(output_dir, 'config')

    try:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"flumy_config_{num_samples}_samples_ntg_{config['ntg']}_max_ch_{config['max_ch_depth']}_isbx_{config['isbx']}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            print(f"\nSaved Flumy configuration to: {config_path}\n")

    except Exception as e:
        print(f"Error saving Flumy configuration: {e}")

def setup_directories(base_dir):
    """Creates the necessary directory structure for saving samples."""
    facies_dir = os.path.join(base_dir, 'samples', 'facies')
    age_dir = os.path.join(base_dir, 'samples', 'age')
    
    os.makedirs(facies_dir, exist_ok=True)
    os.makedirs(age_dir, exist_ok=True)
    
    return facies_dir, age_dir

def flumy_worker(sim_id, base_seed, input_params, facies_dir, age_dir, max_retries=5):
    """
    Worker function to generate a single Flumy simulation with automatic retries.
    Uses the non-expert API parameters: hmax, isbx, ng, zul.
    Extracts the middle block, transposes dimensions to (Z, Y, X), and validates output.
    """
    unique_seed = base_seed + sim_id
    
    # Extract structural params
    nx = input_params['nx']
    ny = input_params['ny']
    mesh = input_params['mesh_size']
    target_nz = input_params['nz']
    bottom_cut = input_params['bottom_cut']
    top_cut = input_params['top_cut']
    dz = input_params['dz']
    
    # Total simulation height = target height + bottom cut + top cut + safety margin
    safety_margin = 10
    zul = target_nz + bottom_cut + top_cut + safety_margin

    print(f"[{unique_seed}] Starting simulation (Seed: {unique_seed}, Zul: {zul}m)")

    for attempt in range(1, max_retries + 1):
        sample_start_time = time.time()
        try:
            # Initialize the simulator
            flsim = Flumy(nx, ny, mesh, verbose=input_params['verbose'])
            
            # Launch simulation
            success = flsim.launch(
                seed=unique_seed,
                hmax=input_params['max_ch_depth'],
                isbx=input_params['isbx'],
                ng=int(input_params['ntg'] * 100), 
                zul=zul
            )
            
            if not success:
                elapsed = time.time() - sample_start_time
                print(f"[{unique_seed}] Flumy internal generation failed after {elapsed:.2f} seconds. (Attempt {attempt}/{max_retries})")
                continue # Skip to next attempt

            # Extract the middle block
            facies, _, age = flsim.getBlock(dz=dz, zb=bottom_cut, nz=target_nz)

            # Replace any 255 values with 1 to ensure valid facies labels
            facies[facies == 255] = 1
            
            # Validating actual physical dimensions from the API (X, Y, Z)
            expected_shape = (nx, ny, target_nz)
            
            if facies.shape != expected_shape:
                elapsed = time.time() - sample_start_time
                print(f"[{unique_seed}] Validation Failed: Expected shape {expected_shape}, got {facies.shape}. (Attempt {attempt}/{max_retries})")
                continue # Skip to next attempt
                
            # --- TRANSPOSE ARRAYS ---
            facies_zyx = np.transpose(facies, (2, 1, 0))
            age_zyx = np.transpose(age, (2, 1, 0))
            
            # File paths
            facies_path = os.path.join(facies_dir, f"sample_{unique_seed}.npy")
            age_path = os.path.join(age_dir, f"sample_{unique_seed}.npy")
            
            # Save the transposed arrays
            np.save(facies_path, facies_zyx)
            np.save(age_path, age_zyx)
            
            # --- FILE VALIDATION BLOCK ---
            all_valid = True
            for filepath in [facies_path, age_path]:
                # 1. OS-level check for completely empty files (0 bytes)
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    all_valid = False
                    print(f"[{unique_seed}] Validation Failed: 0 bytes (Empty file at OS level) - {filepath}")
                    break
                
                # 2. Numpy-level check for corrupted or empty arrays
                try:
                    data = np.load(filepath)
                    if data.size == 0 or data.shape[0] == 0:
                        all_valid = False
                        print(f"[{unique_seed}] Validation Failed: Empty array inside {filepath}")
                        break
                except Exception as e:
                    all_valid = False
                    print(f"[{unique_seed}] Validation Failed: Corrupted/Unreadable by numpy ({str(e)[:40]}...)")
                    break

            # --- OUTCOME HANDLING ---
            if all_valid:
                elapsed = time.time() - sample_start_time
                print(f"[{unique_seed}] Success! Saved transposed shape {facies_zyx.shape}. Time taken: {elapsed:.2f} seconds.")
                return True
            else:
                # Clean up the corrupted files before retrying
                for filepath in [facies_path, age_path]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                print(f"[{unique_seed}] Generation failed validation. Retrying (Attempt {attempt}/{max_retries})...")

        except Exception as e:
            elapsed = time.time() - sample_start_time
            print(f"[{unique_seed}] Exception encountered after {elapsed:.2f} seconds: {e} (Attempt {attempt}/{max_retries})")
            
    print(f"[{unique_seed}] Completely failed to generate valid sample after {max_retries} attempts.")
    return False

def main():
    args = parse_args()

    # --- Configuration ---
    INPUT_PARAMS = {
        'nx': 128,              
        'ny': 128,              
        'mesh_size': 20,        
        'dz': 1,                
        'nz': 32,               
        'bottom_cut': 10,        # Skip the bottom 5m
        'top_cut': 10,           # Skip the top 5m (by only extracting 32m starting from 5m)
        'max_ch_depth': args.max_ch_depth,
        'ntg': args.ntg,            
        'isbx': args.isbx,            
        'verbose': False
    }

    # Parallel & Batch Config
    NUM_SAMPLES = args.num_files
    NUM_WORKERS = args.num_workers
    START_COUNT = 1
    BASE_SEED = 0

    # Determine base path based on OS and set up directories
    base_path = os_check(num_samples=NUM_SAMPLES, ntg=args.ntg, max_ch_depth=args.max_ch_depth, isbx=args.isbx)
    facies_dir, age_dir = setup_directories(base_path)

    # Save the configuration for reproducibility
    save_config(INPUT_PARAMS, NUM_SAMPLES, base_path)

    print(f"Starting Flumy API dataset generation...")
    print(f"Saving outputs to: {base_path}")
    print(f"Workers: {NUM_WORKERS} | Samples: {NUM_SAMPLES}")
    print(f"Target physical dimensions: {INPUT_PARAMS['nx']*INPUT_PARAMS['mesh_size']}m x {INPUT_PARAMS['ny']*INPUT_PARAMS['mesh_size']}m x {INPUT_PARAMS['nz']}m\n")
    
    sample_ids = range(START_COUNT, START_COUNT + NUM_SAMPLES)

    total_start_time = time.time()

    # Spawn workers
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(flumy_worker)(
            sim_id=sid,
            base_seed=BASE_SEED,
            input_params=INPUT_PARAMS,
            facies_dir=facies_dir,
            age_dir=age_dir
        ) for sid in sample_ids
    )

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    success_count = sum(1 for r in results if r)
    
    print("\n" + "="*40)
    print("TEST BATCH COMPLETE")
    print("="*40)
    print(f"Successfully generated and validated: {success_count}/{NUM_SAMPLES} samples.")
    print(f"Total time taken for all samples: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes).")

if __name__ == "__main__":
    main()