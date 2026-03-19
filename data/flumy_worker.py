import os
import subprocess
import time
import numpy as np
import pandas as pd
from flumy_utils import groupFacies, save_sample

def run_flumy_worker(sim_id:int, base_seed:int, flumy_exe_dir, output_dir, temp_dir, save_format='npz', **flumy_params):
    '''
    Worker function ro run  single flumy simulation, optionally post-processing the output and saving it to the dedicated output dir.

    Parameters:
    - sim_id: Unique identifier for the simulation combined with base_seed.
    - base_seed: Manually defined base seed to ensure reproducibility. Final seed: `unique_seed =  base_seed + sim_id`.
    - flumy_exe_dir: Directory where the flumy application is located.
    - output_dir: Output directory for `.npz` files.
    - temp_dir: Temporary dirctory to store the `.f2g` and `.bat` files.
    - save_format: 'npz' (default) or 'h5' for output format.
    - flumy_params: Additional parameters to be passed to the batch file, such as DOMAIN_NX, DOMAIN_NY, etc.

    Returns:
    - A 4D numpy array o
    '''
    unique_seed = base_seed + sim_id
    
    # Define file paths safely
    param_file = os.path.join(temp_dir, f"params_{unique_seed}.bat").replace('\\', '/')
    out_f2g = os.path.join(temp_dir, f"out_{unique_seed}.f2g").replace('\\', '/')
    sample_name = f"sample_{unique_seed}"
    
    # 1. WRITE BATCH FILE DYNAMICALLY
    # We take the base parameters and inject the ones passed from main()
    print(f'[{unique_seed}] Writing batch file: {param_file}')
    
    batch_lines = [
        '[GLOBAL]\n',
        f"VERBOSE = 1\n",
        'F2G_FACIES = 1\n',
        f"F2G_DZ = {flumy_params.get('F2G_DZ',0.5)}\n",
        f"F2G_FILE = {out_f2g}\n",
        '[NEW_SEQ]\n',
        f"SIM_SEED = {unique_seed}\n",
    ]
    
    # Append all other flumy parameters dynamically
    exclude_keys = ['F2G_DZ'] # already handled above
    for key, value in flumy_params.items():
        if key not in exclude_keys:
            batch_lines.append(f"{key} = {value}\n")
            
    with open(param_file, 'w') as f:
        f.writelines(batch_lines)

    # 2. RUN FLUMY
    original_cwd = os.getcwd()
    os.chdir(flumy_exe_dir)
    
    # OS-specific executable call
    if os.name == 'nt':
        exe_cmd = "flumy" # or "flumy.exe", Windows doesn't mind
    else:
        exe_cmd = "./flumy" # The critical Linux prefix
        
    command_text = f'{exe_cmd} -b="{param_file}"'
    
    print(f'[{unique_seed}] Running: {command_text}')
    start_time = time.time()
    
    # Using shell=True is generally okay here, but ensure paths with spaces are quoted if needed
    proc = subprocess.Popen(command_text, shell=True)
    proc.wait()
    
    os.chdir(original_cwd) # Revert working directory
    print(f"[{unique_seed}] Simulation finished in {np.round((time.time() - start_time)/60,2)} minutes.")

    # 3. POST-PROCESS
    try:
        if os.path.exists(out_f2g):
            # Parse F2G file
            df = pd.read_csv(out_f2g, sep=r'\s+', comment='F', header=None)
            
            # Bulletproof casting
            raw_col = pd.to_numeric(df.iloc[:, 0], errors='coerce') 
            raw_col = raw_col.fillna(0).values 
            facies_1d = np.clip(raw_col, 0, 255).astype(np.int32)
            
            # Dimensions setup (pulled from flumy_params)
            nx = flumy_params.get('DOMAIN_NX', 128)
            ny = flumy_params.get('DOMAIN_NY', 128)
            nz_target = flumy_params.get('ZUL_TOPO', 64) 
            
            total_valid_cells = (len(facies_1d) // (nx * ny)) * (nx * ny)
            clean_facies_1d = facies_1d[:total_valid_cells]
            
            # Reshape
            actual_nz = total_valid_cells // (nx * ny)
            facies_3d_temp = clean_facies_1d.reshape((actual_nz, ny, nx))
            
            # Crop to target Z
            facies_3d = facies_3d_temp[:nz_target, :, :] 
            
            # Group facies (One-hot encoding is DISABLED here as requested)
            facies_grouped = facies_3d#groupFacies(facies_3d)
            
            # Save data
            data_dict = {'facies': facies_grouped}
            saved_file = save_sample(data_dict, output_dir, sample_name, save_format)
            
            print(f"[{unique_seed}] Saved training sample: {saved_file}")
            
            # Cleanup temp files
            os.remove(param_file)
            os.remove(out_f2g)
            return True
        else:
            print(f"[{unique_seed}] Error: {out_f2g} was not created.")
            return False
            
    except Exception as e:
        print(f"[{unique_seed}] Processing Error: {e}")
        return False