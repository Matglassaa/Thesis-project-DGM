import os
import subprocess
import numpy as np
import pandas as pd 
import stat
import time

def groupFacies(fac, grouping_scheme=None):
    """
    Function to group all facies produced by the flumy simulation into three main categories: FA1, FA2 and FA3.
    - FA1: Fluvial and distributary channel deposits
    - FA2: Crevasse splays and levee deposits
    - FA3: Overbank deposists and paleosols

    ### Args:
        fac (numpy array): 3D array of facies produced by the flumy simulation:
        n_groups (int, optional): Number of groups to create. Defaults to 3.
        grouping_scheme (list or dict, optional): Scheme to use for grouping the facies. If None, the default scheme is used. Defaults to None.

    ### Returns:
        fac_grouped (numpy array): 3D array of grouped facies

    ### Examples:
        **Example 1:** Using the default grouping scheme
            - 0:4 -> 1 (FA1: Fluvial and distributary channel deposits)
            - 4:8 -> 7 (FA2: Crevasse splays and levee deposits)
            - 8:13 -> 8 (FA3: Overbank deposists and paleosls)

            
        **Example 2:** Using a custom grouping scheme
            - grouping_scheme = {1: [1], 2: [2,3,4], 7: [5,6,7], 8: [8,9,10,11,12,13]}
            - 1 -> (FA1: Channel Lag)
            - 2:4 -> (FA1: Channel Fill)
            - 5:7 -> (FA2: Crevasse splays and levee deposits)
            - 8:13 -> (FA3: Overbank deposists and paleosls)
    """
    fac_int = fac.astype(np.int32)
    lut = np.zeros(256, dtype=np.uint8)

    if grouping_scheme is None:
        # DEFAULT SCHEME
        lut[0:4] = 1       # FA1: Fluvial and distributary channel deposits 
        lut[4:8] = 7       # FA2: Crevasse splays and levee deposits
        lut[8:13] = 8      # FA3: Overbank deposists and paleosols
        
    else:
        # CUSTOM SCHEME
        # e.g., grouping_scheme = {1: [0,1,2,3], 7: [4,5,6,7], 8: [8,9,10,11,12,13]}
        for new_val, old_vals in grouping_scheme.items():
            # Convert old_vals to a list to use as indices
            lut[list(old_vals)] = new_val

    return lut[fac_int]

def one_hot_encode_3d(fac, categories=[1, 7, 8], channel_first=True):
    """
    Converts a 3D categorical array into a 4D one-hot encoded array.
    
    Args:
        fac (numpy.ndarray): 3D array of grouped facies.
        categories (list): The specific integer values to encode. 
                           Default matches the 1, 7, 8 from groupFacies.
        channel_first (bool): If True, returns shape (Channels, Z, Y, X) for PyTorch.
                              If False, returns shape (Z, Y, X, Channels) for TensorFlow.
    """
    # This creates a boolean array by checking fac against each category, 
    # then converts True/False to 1/0 (uint8 to save memory)
    one_hot = (fac[..., None] == categories).astype(np.uint8)
    
    if channel_first:
        # Move the newly created channel dimension from the end to the front
        one_hot = np.moveaxis(one_hot, -1, 0)
        
    return one_hot

def worker_flumy_8501(sim_id, base_seed, output_dir, temp_dir):
    unique_seed = base_seed + sim_id
    
    # 1. SETUP CLEAN PATHS
    
    # safe_temp_root = "C:/flumy_temp" 
    # if not os.path.exists(safe_temp_root):
    #     os.makedirs(safe_temp_root)
    
    # # Grant full permissions to this folder
    # os.chmod(safe_temp_root, stat.S_IWRITE)

    flumy_exe_dir = r"C:\Users\mathi\Downloads\flumy_8.501_win64\bin"
    
    # Define files using the safe root
    param_file = temp_dir + "/params_" + str(unique_seed) + ".bat"
    out_f2g = temp_dir + "/out_" + str(unique_seed) + ".f2g"
    final_npz = os.path.abspath(output_dir).replace('\\', '/') + "/sample_" + str(unique_seed) + ".npz"

    # 2. WRITE BATCH (Ensuring no trailing spaces and clean quotes)
    print(f'Writing batch file to safe location: {param_file}')
    with open(param_file, 'w') as f:
        f.writelines([
            '[GLOBAL]\n',
            'VERBOSE = 1\n',
            'F2G_DZ = 2.5\n',
            'F2G_FACIES = 1\n',
            'F2G_GRAIN = 0\n',
            'F2G_ORDER = +X +Y +Z\n',
            'F2G_FILE = ' + out_f2g + '\n', 
            '[NEW_SEQ]\n',
            'SIM_SEED = ' + str(unique_seed) + '\n',
            'DOMAIN_NX = 100\n',
            'DOMAIN_NY = 50\n',
            'DOMAIN_DX = 20\n',
            'DOMAIN_DY = 20\n',
            'ZUL_TOPO = 60\n',
            'ZUL_TYPE = 2\n',
            'AG_TYPE = 2\n',
            'AG_OB_FREQ = 1\n',
            'AG_OB_PERIOD = 100\n',
            'AG_OB_MAX = 0.8\n',
            'CHNL_WIDTH = 40\n',
            'CHNL_MAX_DEPTH = 4\n',
            'CHNL_FLW_DIR = 210\n',
            'LAUNCH_IT = -1\n'
        ])

    # 3. RUN BATCH
    os.chdir(flumy_exe_dir)
    # Wrap param_file in quotes for the command line
    command_text = f'flumy -b="{param_file}"'
    
    print(f'Running: {command_text}')
    start_time = time.time()
    
    # Using Popen as requested by your original simple example
    proc = subprocess.Popen(command_text, shell=True)
    proc.wait()
    
    print("Simulation finished in --- %s minutes ---" % (np.round((time.time() - start_time)/60,2)))

    # 4. POST-PROCESS
    # 4. POST-PROCESS
    try:
        if os.path.exists(out_f2g):
            # 1. Read the file, skipping the header lines robustly
            df = pd.read_csv(out_f2g, sep=r'\s+', comment='F', header=None)
            
            # 2. BULLETPROOF CASTING: Handle NaNs/Text before converting to integer
            raw_col = pd.to_numeric(df.iloc[:, 0], errors='coerce') # Turns text to NaN
            raw_col = raw_col.fillna(0).values                      # Turns NaN to 0
            facies_1d = np.clip(raw_col, 0, 255).astype(np.int32)   # Safe to cast now!
            
            # 3. Calculate dimensions
            nx, ny, nz_target = 100, 50, 20
            total_valid_cells = (len(facies_1d) // (nx * ny)) * (nx * ny)
            clean_facies_1d = facies_1d[:total_valid_cells]
            
            # Reshape to 3D (Z, Y, X)
            actual_nz = total_valid_cells // (nx * ny)
            facies_3d_temp = clean_facies_1d.reshape((actual_nz, ny, nx))
            
            # 4. CROP TO BOTTOM 20 LAYERS
            # Because you set ZUL_TOPO=60, the bottom 50m (20 layers of 2.5m) 
            # is 100% solid sediment with no empty "air" cells.
            facies_3d = facies_3d_temp[:nz_target, :, :] 
            
            # 5. Group and Encode
            facies_grouped = groupFacies(facies_3d)
            facies_one_hot = one_hot_encode_3d(facies_grouped)
            
            np.savez_compressed(final_npz, facies=facies_one_hot)
            print(f"✅ Saved training sample (Cropped bottom {nz_target} layers): {final_npz}")
            
            # Cleanup
            os.remove(param_file)
            os.remove(out_f2g)
            return True
        else:
            print(f"❌ Error: {out_f2g} was not created. Permission still denied?")
            return False
            
    except Exception as e:
        print(f"❌ Processing Error: {e}")
        return False

if __name__ == "__main__":
    # 1. Define your test parameters
    test_base_seed = 42
    test_sim_id = 1
    
    # 2. Define where you want the temporary and final files to go
    base_path = os.path.join(os.getcwd(),'data')
    output_directory = os.path.join(base_path,"test_outputs")
    temp_directory = os.path.join(base_path,"test_temp")
    
    # Create the directories if they don't exist yet
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(temp_directory, exist_ok=True)
    
    # 3. Call the worker function to run the simulation
    print(f"Launching FLUMY test run (Seed: {test_base_seed + test_sim_id})...")
    
    success = worker_flumy_8501(
        sim_id=test_sim_id, 
        base_seed=test_base_seed, 
        output_dir=output_directory, 
        temp_dir=temp_directory
    )
    
    # 4. Verify the results
    if success:
        print("\n✅ Simulation completed successfully!")
        
        # Check if the generated .npz file exists
        expected_npz_path = os.path.join(output_directory, f"sample_{test_base_seed + test_sim_id}.npz")
        
        if os.path.exists(expected_npz_path):
            print(f"Loading generated file: {expected_npz_path}")
            
            # Load the compressed data to verify it worked
            data = np.load(expected_npz_path)
            facies_tensor = data['facies']
            
            print("\n--- Tensor Verification ---")
            print(f"Tensor Shape: {facies_tensor.shape}") 
            print(f"Data Type: {facies_tensor.dtype}")
            print(f"Unique Values: {np.unique(facies_tensor)}")
            
        else:
            print("❌ Error: The worker reported success, but the .npz file is missing.")
    else:
        print("\n❌ Simulation failed. Check your FLUMY path and parameters.")