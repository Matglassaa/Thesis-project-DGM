import os
import subprocess
import time
import numpy as np
import pandas as pd
from flumy_utils import groupFacies, save_sample


def batch_writer(param_file, out_f2g, unique_seed, **flumy_params):
    """
    Generates the .bat configuration file for the Flumy executable.
    
    Parameters:
    - param_file (str): The pth to the .bat file.
    - out_f2g (str): The path to the output .f2g file.
    - unique_seed (int): A unique seed to ensure reproducibility.
    
    Returns:
    - None
    """
    # 1. Create intial layout of .bat file in list format
    batch_lines = [
        '[GLOBAL]\n', f"VERBOSE = 0\n", 'F2G_FACIES = 1\n',
        f"F2G_DZ = {flumy_params.get('F2G_DZ')}\n",
        f"F2G_FILE = {out_f2g}\n",
        '[NEW_SEQ]\n', f"SIM_SEED = {unique_seed}\n",
    ]

    exclude_keys = ['F2G_DZ']       # Make sure to exclude F2G_DZ as this param was already defined above

    # 2. Loop over all parameters ni `flumy_params` and add them to batch_lines file
    for key, value in flumy_params.items():
        if key not in exclude_keys:
            batch_lines.append(f"{key} = {value}\n")

    # 3. Write .bat file using batch_lines list        
    with open(param_file, 'w') as f:
        f.writelines(batch_lines)


def run_flumy_executable(flumy_exe_dir, param_file, unique_seed):
    """
    Handles the subprocess call and directory switching.
    
    Parameters:
    - flumy_exe_dir (str): The path to the flumy executable directory.
    - param_file (str): The pth to the .bat file.
    - unique_seed (int): A unique seed to ensure reproducibility.

    Returns:
    - None
    """
    # 1. Save currect working directory and save as variable
    original_cwd = os.getcwd()

    # 2. 'Try' block for running the Flumy excecutable 
    try:
        os.chdir(flumy_exe_dir)
        exe_cmd = "flumy" if os.name == 'nt' else "./flumy"         # Automatically detect whether you are wokring in Linux or Windows
        command_text = f'{exe_cmd} -b="{param_file}"'               # Input all params into flumy excecutable (.f2g is saved there saved in that process well)
        
        print(f'[{unique_seed}] Running: {command_text}')
        start_time = time.time()
        
        # 2.1. Open the process in a subprocess which allows for multi-process simulation.
        proc = subprocess.Popen(command_text)           # Flexibly execute a command in a new process
        proc.wait()
        
        print(f"[{unique_seed}] Simulation finished in {np.round((time.time() - start_time)/60, 2)} min.")
    # 3. IMPORTANT: No matter what happens, always return to the orginal working directory 
    finally:
        os.chdir(original_cwd)


def parse_flumy_output(out_f2g, flumy_params, top_slice_offset):
    """
    Reads the .f2g file and converts it into a structured 3D numpy array.
    
    Parameters:
    - out_f2g (str): The path to the output .f2g file.
    - flumy_params (dict): A dictionary of all parameters modified to spawn the Flumy process

    Returns:
    - np.ndarray: A 3D array of facies
    """
    # 1. Check for path existance & read .csv output file
    if not os.path.exists(out_f2g):
        raise FileNotFoundError(f"Output file {out_f2g} was not created.")
    #df = pd.read_csv(out_f2g, sep=r'\s+', comment='F', header=None)
    raw_data = np.genfromtxt(out_f2g, comments='F', missing_values='NaN', filling_values=0)

    #raw_col = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values 
    raw_col = np.nan_to_num(raw_data, nan=0.0)

    #facies_1d = np.clip(raw_col, 0, 255).astype(np.int32)                       # Limit x,y,z grid values to a range between 0 and 255 (256 options in total)
    facies_1d = np.clip(raw_col, 0, 255).astype(np.int32)
    
    # 2. Get domain values from input
    nx = flumy_params.get('DOMAIN_NX', 256)
    ny = flumy_params.get('DOMAIN_NY', 256)
    zul_topo = flumy_params.get('ZUL_TOPO', 64) 
    dz = flumy_params.get('F2G_DZ', 1.0)
    
    #3. Calculate height of full voxel grid and convert to a 3D grid
    actual_nz = len(facies_1d) // (nx * ny)
    facies_3d_temp = facies_1d[:actual_nz * nx * ny].reshape((actual_nz, ny, nx))
    nz_target = int(zul_topo / dz)
    
    # 4. Slice logic (keeping your specific slice -10) to end up with right dimensions
    if actual_nz >= nz_target:
        return facies_3d_temp[-nz_target:-top_slice_offset, :, :]
    else:
        print(f"Warning: actual_nz ({actual_nz}) < nz_target ({nz_target})")
        return facies_3d_temp


def run_flumy_worker(sim_id, base_seed, flumy_exe_dir, output_dir, temp_dir, top_slice_offset=10, **flumy_params):
    """
    The main orchestrator function.
    
    Parameters:
    - sim_id: id which belongs to current simulation -> makes sure each simulation has an unique id.
    - base_seed: base seed of the simulation.
    - flumy_exe_dir (str): The path to the flumy executable directory.
    - output_dir: Path towards exact output directory.
    - temp_dir: Path towards temporary directory to store .bat and .f2g input files.
    - top_slice_offset (int): Indicate the amounnt of voxels that should be taken off the maximum height -> this limits full channels to be present in the final 3D grid.
    - flumy_params (dict): A dictionary of all parameters modified to spawn the Flumy process

    Returns:
    - None
    """
    # 1. Define unique seed & path towards input- and output file.
    unique_seed = base_seed + sim_id
    param_file = os.path.join(temp_dir, f"params_{unique_seed}.bat").replace('\\', '/')
    out_f2g = os.path.join(temp_dir, f"out_{unique_seed}.f2g").replace('\\', '/')
    
    # 2. Run all steps to generate ONE sample
    try:
        # 2.1. Setup
        batch_writer(param_file, out_f2g, unique_seed, **flumy_params)
        
        # 2.2. Execute
        run_flumy_executable(flumy_exe_dir, param_file, unique_seed)
        
        # 2.3. Process
        facies_3d = parse_flumy_output(out_f2g, flumy_params, top_slice_offset)
        
        # 2.4. Save
        save_sample({'facies': facies_3d}, output_dir, f"sample_{unique_seed}", 'npz')
        
        return True
    except Exception as e:
        print(f"[{unique_seed}] Critical Error: {e}")
        return False
    
    # 5. Cleanup (Always runs even if an error occurs)
    finally:
        for f in [param_file, out_f2g]:
            if os.path.exists(f):
                os.remove(f)