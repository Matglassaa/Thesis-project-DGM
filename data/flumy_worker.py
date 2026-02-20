import os
import numpy as np
from flumy import Flumy
from flumy_utils import FlumyDataManager

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

    return lut[fac]

def run_simulation_batch(batch_id, n_samples_in_batch, grid_params, sim_params, output_dir, show_cores= False):
    """
    Worker function for generating each sample.

    Args:
        batch_id (int): Unique identifier for the batch.
        n_samples_in_batch (int): Number of samples to generate in this batch.
        grid_params (dict): Dictionary containing grid parameters (nx, ny, mesh, nz).
        sim_params (dict): Dictionary containing simulation parameters (base_seed, max_channel_depth, etc.).
        output_dir (str): Directory where the output HDF5 file will be saved.
    """
    manager = FlumyDataManager(output_dir=output_dir)

    if show_cores:
        pid = os.getpid()
        print(f"--> Batch {batch_id} is running on Core ID: {pid}")
    
    h5_path = manager.initialize_h5_file(
        batch_id, 
        n_samples_in_batch, 
        grid_params['nx'], 
        grid_params['ny'], 
        grid_params['nz']
    )
    
    for i in range(n_samples_in_batch):
        flsim = Flumy(grid_params['nx'], grid_params['ny'], grid_params['mesh'], verbose=False)
        
        unique_seed = sim_params['base_seed'] + (batch_id * 10000) + i
        
        flsim.launch(
            unique_seed, 
            hmax=sim_params['max_channel_depth'], 
            isbx=sim_params['max_sand_body_extention'], 
            ng=sim_params['net_gross'], 
            zul=sim_params['target_height'], 
            niter=sim_params['niter'],
            lvb=sim_params['levee_break']
        )
        
        fac, grain, age = flsim.getBlock(dz=sim_params['vertical_resolution'], zb=0, nz=grid_params['nz'])
        fac = groupFacies(fac)
        manager.save_to_batch(h5_path, index=i, fac=fac, grain=grain, age=age)
        
    return h5_path