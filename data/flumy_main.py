import os
import time
from joblib import Parallel, delayed
from flumy_worker import run_flumy_worker

def main():
    FLUMY_EXE_DIR = r"C:\Users\mathi\Downloads\flumy_8.501_win64\bin"
    
    BASE_PATH = os.path.join(os.getcwd(), 'data')
    OUTPUT_DIR = os.path.join(BASE_PATH, "test_outputs")
    TEMP_DIR = os.path.join(BASE_PATH, "test_temp")
    
    NUM_SAMPLES = 5
    BASE_SEED = 42
    
    # Execution options
    USE_PARALLEL = True
    N_JOBS = -1  # -1 uses all available CPU cores
    SAVE_FORMAT = 'npz' # change to 'h5' if needed
    
    FLUMY_PARAMS = {
        'F2G_DZ': 0.5,
        'DOMAIN_NX': 128,
        'DOMAIN_NY': 128,
        'DOMAIN_DX': 20,
        'DOMAIN_DY': 20,
        'ZUL_TOPO': 64,
        'ZUL_TYPE': 1,
        'CHNL_WIDTH': 60,
        'CHNL_MAX_DEPTH': 3,
        'CHNL_WAVELENGTH': 600,
        'AV_REG_FREQ': 2,
        'AV_REG_POISSON': 800,
        'AV_LOC_FREQ': 2,
        'AV_LOC_POISSON': 500,
        'AV_LV_OB': 1,
        'AG_TYPE': 2,
        'AG_OB_FREQ': 2,
        'AG_OB_POISSON': 500,
        'AG_OB_DIST': 3,
        'AG_OB_LOGNORM_MEAN': 0.05,
        'AG_OB_LOGNORM_STDEV': 0.01,
        'LAUNCH_IT': -1
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    sample_ids = list(range(1, NUM_SAMPLES + 1))
    
    print(f"Starting batch of {NUM_SAMPLES} samples.")
    print(f"Parallel: {USE_PARALLEL} | Format: .{SAVE_FORMAT}")
    
    start_time = time.time()
    
    if USE_PARALLEL:
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_flumy_worker)(
                sim_id=sid,
                base_seed=BASE_SEED,
                flumy_exe_dir=FLUMY_EXE_DIR,
                output_dir=OUTPUT_DIR,
                temp_dir=TEMP_DIR,
                save_format=SAVE_FORMAT,
                **FLUMY_PARAMS
            ) for sid in sample_ids
        )
    else:
        results = []
        for sid in sample_ids:
            res = run_flumy_worker(
                sim_id=sid,
                base_seed=BASE_SEED,
                flumy_exe_dir=FLUMY_EXE_DIR,
                output_dir=OUTPUT_DIR,
                temp_dir=TEMP_DIR,
                save_format=SAVE_FORMAT,
                **FLUMY_PARAMS
            )
            results.append(res)
            
    end_time = time.time()
    success_count = sum(1 for r in results if r)
    
    print("\n" + "="*40)
    print("BATCH GENERATION COMPLETE")
    print("="*40)
    print(f"Successfully generated: {success_count}/{NUM_SAMPLES} samples.")
    print(f"Total time taken: {(end_time - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()