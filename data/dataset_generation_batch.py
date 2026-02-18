# import os
# import joblib
# import numpy as np
# from flumy import Flumy 
# from flumy_utils import FlumyDataManager, BatchSimulator

# if __name__ == "__main__":
    
#     # Simulation Config
#     TOTAL_SAMPLES = 12
#     BATCH_SIZE = 5  # Samples per H5 file (Result: 200 files)
#     N_JOBS = -1      # Use all cores
    
#     OUTPUT_DIR = os.path.join("data", "datasets", "training")
    
#     # Grid Parameters
#     grid_params = {
#         'nx': 256, 'ny': 256, 'mesh': 18, 
#         'nz': int(10 / 0.5) # calculated based on height/resolution
#     }
    
#     # Flumy Hyperparameters
#     sim_params = {
#         'base_seed': 42,
#         'max_channel_depth': 5,
#         'net_gross': 70,
#         'max_sand_body_extention': 40,
#         'target_height': 10,
#         'vertical_resolution': 0.5,
#         'niter': -1
#     }

#     # Initialize Simulator
#     # We pass the 'run_simulation_batch' function to the simulator
#     simulator = BatchSimulator(
#         total_samples=TOTAL_SAMPLES,
#         batch_size=BATCH_SIZE,
#         grid_params=grid_params,
#         sim_params=sim_params,
#         n_jobs=N_JOBS
#     )

#     # Run Parallel Generation
#     print(f"Generating {TOTAL_SAMPLES} samples in batches of {BATCH_SIZE}...")
#     output_files = simulator.run(grid_params, sim_params, OUTPUT_DIR)
    
#     print(f"Done! Generated {len(output_files)} batch files.")

import os
from flumy_utils import BatchSimulator

if __name__ == "__main__":
    
    # --- Configuration ---
    TOTAL_SAMPLES = 12
    BATCH_SIZE = 5
    N_JOBS = -1
    OUTPUT_DIR = os.path.join("data", "datasets", "training")
    
    # Grid Parameters
    grid_params = {
        'nx': 256, 'ny': 256, 'mesh': 18, 
        'nz': int(10 / 0.5)
    }
    
    # Flumy Hyperparameters
    sim_params = {
        'base_seed': 42,
        'max_channel_depth': 5,
        'net_gross': 70,
        'max_sand_body_extention': 40,
        'target_height': 10,
        'vertical_resolution': 0.5,
        'niter': -1
    }

    # --- Initialization ---
    # Note: We pass all params here now, making the class self-contained
    simulator = BatchSimulator(
        total_samples=TOTAL_SAMPLES,
        batch_size=BATCH_SIZE,
        grid_params=grid_params,
        sim_params=sim_params,
        output_dir=OUTPUT_DIR,
        n_jobs=N_JOBS
    )

    # --- Execution ---
    print(f"Generating {TOTAL_SAMPLES} samples...")
    output_files = simulator.run()
    
    print(f"Done! Generated {len(output_files)} batch files.")