import os
import subprocess
import time
import numpy as np
import pandas as pd
from flumy_utils import groupFacies, save_sample


def batch_writer(param_file, out_f2g, flow_dir, unique_seed, **flumy_params):
    # create batch lines
    batch_lines = [
        '[GLOBAL]\n', 
        f"VERBOSE = 0\n", 
        'F2G_FACIES = 1\n',
        'F2G_GRAIN = 1\n',
        'F2G_AGE = 1\n',
        f"F2G_DZ = {flumy_params.get('F2G_DZ')}\n",
        f"F2G_FILE = {out_f2g}\n",
        '[NEW_SEQ]\n', 
        f"SIM_SEED = {unique_seed}\n",
        f"CHNL_FLW_DIR = {flow_dir}\n"
    ]

    exclude_keys = ['F2G_DZ']       
    for key, value in flumy_params.items():
        if key not in exclude_keys:
            batch_lines.append(f"{key} = {value}\n")
      
    with open(param_file, 'w') as f:
        f.writelines(batch_lines)


def run_flumy_executable(flumy_exe_dir, param_file, unique_seed):
    original_cwd = os.getcwd()
    try:
        os.chdir(flumy_exe_dir)
        exe_cmd = "flumy.exe" if os.name == 'nt' else "./flumy" 
        
        command_list = [exe_cmd, f"-b={param_file}"] 
        
        print(f'[{unique_seed}] Running: {" ".join(command_list)}')
        start_time = time.time()
        
        full_exe_path = os.path.join(flumy_exe_dir, "flumy")
        proc = subprocess.Popen([full_exe_path, f"-b={param_file}"])          
        proc.wait()
        
        print(f"[{unique_seed}] Simulation finished in {np.round((time.time() - start_time)/60, 2)} min.")
        
    finally:
        os.chdir(original_cwd)


def parse_flumy_output(out_f2g, flumy_params, top_slice_offset):
    if not os.path.exists(out_f2g):
        raise FileNotFoundError(f"Output file {out_f2g} was not created.")
    
    # Read the data
    raw_data = np.genfromtxt(out_f2g, comments='F', missing_values='NaN', filling_values=0)
    
    # 1. Catch the empty file scenario immediately
    if raw_data.size == 0:
        raise ValueError(f"The .f2g output file is empty. Flumy likely crashed internally.")
        
    # 2. Safely split the columns based on array dimensions
    if raw_data.ndim == 1:
        print("Warning: Only 1 column found in .f2g file. Defaulting grain/age to 0.")
        raw_facies = raw_data
        raw_grain = np.zeros_like(raw_facies)
        raw_age = np.zeros_like(raw_facies)
    else:
        raw_facies = raw_data[:, 0]
        raw_grain = raw_data[:, 1] if raw_data.shape[1] >= 2 else np.zeros_like(raw_facies)
        raw_age = raw_data[:, 2] if raw_data.shape[1] >= 3 else np.zeros_like(raw_facies)

    # Clean and set types 
    facies_1d = np.clip(np.nan_to_num(raw_facies, nan=0.0), 0, 255).astype(np.int32)
    grain_1d = np.nan_to_num(raw_grain, nan=0.0).astype(np.float32)
    age_1d = np.nan_to_num(raw_age, nan=0.0).astype(np.float32)
    
    # 2. Get domain values
    nx = flumy_params.get('DOMAIN_NX', 256)
    ny = flumy_params.get('DOMAIN_NY', 256)
    zul_topo = flumy_params.get('ZUL_TOPO', 64) 
    dz = flumy_params.get('F2G_DZ', 1.0)
    
    # 3. Calculate height and reshape ALL THREE arrays
    actual_nz = len(facies_1d) // (nx * ny)
    grid_shape = (actual_nz, ny, nx)
    
    facies_3d_temp = facies_1d[:actual_nz * nx * ny].reshape(grid_shape)
    grain_3d_temp = grain_1d[:actual_nz * nx * ny].reshape(grid_shape)
    age_3d_temp = age_1d[:actual_nz * nx * ny].reshape(grid_shape)
    
    nz_target = int(zul_topo / dz)
    
    # 4. Slice arrays using your specific top_slice_offset
    if actual_nz >= nz_target:
        return (
            facies_3d_temp[-nz_target:-top_slice_offset, :, :], 
            grain_3d_temp[-nz_target:-top_slice_offset, :, :],
            age_3d_temp[-nz_target:-top_slice_offset, :, :]
        )
    else:
        print(f"Warning: actual_nz ({actual_nz}) < nz_target ({nz_target})")
        return facies_3d_temp, grain_3d_temp, age_3d_temp


def run_flumy_worker(sim_id, base_seed, flumy_exe_dir, output_dir, temp_dir, top_slice_offset=10, save_format='npz', save_all_data=False, max_retries=5, **flumy_params):
    unique_seed = base_seed + sim_id
    param_file = os.path.join(temp_dir, f"params_{unique_seed}.bat").replace('\\', '/')
    out_f2g = os.path.join(temp_dir, f"out_{unique_seed}.f2g").replace('\\', '/')
    flow_dir = np.random.uniform(low=85, high=95)
    
    print(f"[{unique_seed}] Starting simulation with seed {unique_seed} and flow direction {flow_dir}.")
    
    # --- AUTOMATIC RETRY LOOP ---
    for attempt in range(1, max_retries + 1):
        try:
            batch_writer(param_file, out_f2g, flow_dir=flow_dir, unique_seed=unique_seed, **flumy_params)
            run_flumy_executable(flumy_exe_dir, param_file, unique_seed)
            
            # Unpack facies, grain, and age 3D arrays here
            facies_3d, grain_3d, age_3d = parse_flumy_output(out_f2g, flumy_params, top_slice_offset)
            
            if save_all_data:
                data_dict = {
                    'facies': facies_3d,
                    'grain': grain_3d,
                    'age': age_3d
                }
            else:
                data_dict = {
                    'facies': facies_3d
                }
            
            # Pass the dictionary and format down to your utils
            saved_paths = save_sample(data_dict, output_dir, f"sample_{unique_seed}", save_format=save_format)
            
            # Ensure saved_paths is a list 
            if isinstance(saved_paths, str):
                saved_paths = [saved_paths]
                
            # --- VALIDATION BLOCK ---
            all_valid = True
            
            for filepath in saved_paths:
                # 1. OS-level check for completely empty files (0 bytes)
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    all_valid = False
                    print(f"[{unique_seed}] Validation Failed: 0 bytes (Empty file at OS level) - {filepath}")
                    break
                
                # 2. Numpy-level check for corrupted or empty arrays
                try:
                    if filepath.endswith('.npz'):
                        with np.load(filepath) as data:
                            if 'facies' not in data:
                                all_valid = False
                                print(f"[{unique_seed}] Validation Failed: Missing 'facies' array key in {filepath}")
                                break
                            elif data['facies'].size == 0 or data['facies'].shape[0] == 0:
                                all_valid = False
                                print(f"[{unique_seed}] Validation Failed: Empty 'facies' array inside {filepath}")
                                break
                    elif filepath.endswith('.npy'):
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
                return True  # Success! Exit the retry loop.
            else:
                # Clean up the corrupted files before retrying
                for filepath in saved_paths:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                print(f"[{unique_seed}] Generation failed validation. Retrying (Attempt {attempt}/{max_retries})...")
                
        except Exception as e:
            print(f"[{unique_seed}] Critical Error on attempt {attempt}/{max_retries}: {e}")
            
        finally:
            # Always clean up temporary Flumy I/O files before the next attempt
            for f in [param_file, out_f2g]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
                        
    print(f"[{unique_seed}] Completely failed to generate valid sample after {max_retries} attempts.")
    return False