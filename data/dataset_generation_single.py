import os
from flumy_utils import *
from flumy_worker import *

if __name__ == "__main__":
    TOTAL_SAMPLES = 1                                              # set to 10000 for full parallelization on TU Delft cluster
    BATCH_SIZE = 1                                                  # set to 25   "                                            "
    N_JOBS = 1                                                   # Switch to 128 "                                          "
    OUTPUT_DIR = os.path.join("data", "datasets", "training")
    
    # Grid Parameters
    grid_params = {
        'nx': 256, 'ny': 256, 'mesh': 18, 
        'nz': int(10 / 0.5)
    }
    
    # Flumy Hyperparameters
    sim_params = {
        'base_seed': 42,
        'max_channel_depth': 3,
        'net_gross': 70,
        'max_sand_body_extention': 40,
        'target_height': 10,
        'vertical_resolution': 0.5,
        'niter': -1,
        'levee_break':True
    }

    # --- Initialization ---
    # Note: We pass all params here now, making the class self-contained
    simulator = BatchSimulator(
        worker_function=run_simulation_batch,
        total_samples=TOTAL_SAMPLES,
        batch_size=BATCH_SIZE,
        grid_params=grid_params,
        sim_params=sim_params,
        output_dir=OUTPUT_DIR,
        n_jobs=N_JOBS
    )

    # --- Execution ---
    print(f"Generating {TOTAL_SAMPLES} samples")
    output_files = simulator.run()
    
    print(f"Done! Generated {len(output_files)} batch files.")