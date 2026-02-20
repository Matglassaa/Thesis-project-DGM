import os
import h5py
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

# def run_simulation_batch(batch_id, n_samples_in_batch, grid_params, sim_params, output_dir, show_cores= False):
#     """
#     Worker function for generating each sample.

#     Args:
#         batch_id (int): Unique identifier for the batch.
#         n_samples_in_batch (int): Number of samples to generate in this batch.
#         grid_params (dict): Dictionary containing grid parameters (nx, ny, mesh, nz).
#         sim_params (dict): Dictionary containing simulation parameters (base_seed, max_channel_depth, etc.).
#         output_dir (str): Directory where the output HDF5 file will be saved.
#     """
#     manager = FlumyDataManager(output_dir=output_dir)

#     if show_cores:
#         pid = os.getpid()
#         print(f"--> Batch {batch_id} is running on Core ID: {pid}")
    
#     h5_path = manager.initialize_h5_file(
#         batch_id, 
#         n_samples_in_batch, 
#         grid_params['nx'], 
#         grid_params['ny'], 
#         grid_params['nz']
#     )
    
#     for i in range(n_samples_in_batch):
#         flsim = Flumy(grid_params['nx'], grid_params['ny'], grid_params['mesh'], verbose=False)
        
#         unique_seed = sim_params['base_seed'] + (batch_id * 10000) + i
        
#         flsim.launch(
#             unique_seed, 
#             hmax=sim_params['max_channel_depth'], 
#             isbx=sim_params['max_sand_body_extention'], 
#             ng=sim_params['net_gross'], 
#             zul=sim_params['target_height'], 
#             niter=sim_params['niter'],
#             lvb=sim_params['levee_break']
#         )
        
#         fac, grain, age = flsim.getBlock(dz=sim_params['vertical_resolution'], zb=0, nz=grid_params['nz'])
#         manager.save_to_batch(h5_path, index=i, fac=fac, grain=grain, age=age)
        
#     return h5_path

class FlumyDataManager:
    def __init__(self, output_dir="datasets/training"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize_h5_file(self, batch_id, n_samples, nx, ny, nz):
        filename = f"flumy_batch_{batch_id:04d}.h5"
        file_path = os.path.join(self.output_dir, filename)
        
        with h5py.File(file_path, 'w') as f:
            chunks = (1, nx, ny, nz)
            f.create_dataset("facies", (n_samples, nx, ny, nz), dtype='u1', compression="gzip", chunks=chunks)
            f.create_dataset("grain", (n_samples, nx, ny, nz), dtype='f4', compression="gzip", chunks=chunks)
            f.create_dataset("age", (n_samples, nx, ny, nz), dtype='i4', compression="gzip", chunks=chunks)
            
            f.attrs['n_samples'] = n_samples
            f.attrs['batch_id'] = batch_id
        
        return file_path

    def save_to_batch(self, file_path, index, fac, grain, age):
        with h5py.File(file_path, 'a') as f:
            f['facies'][index, ...] = fac.astype(np.uint8)
            f['grain'][index, ...] = grain.astype(np.float32)
            f['age'][index, ...] = age.astype(np.int32)

class BatchSimulator:
    def __init__(self, worker_function, total_samples, batch_size, grid_params, sim_params, output_dir, n_jobs=-1):
        self.worker = worker_function
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.grid_params = grid_params
        self.sim_params = sim_params
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        """
        Orchestrates the batches.
        """
        n_cores_active = effective_n_jobs(self.n_jobs)
        n_batches = int(np.ceil(self.total_samples / self.batch_size))

        jobs = []
        for i in range(n_batches):
            batch_id = i
            current_batch_size = min(
                self.batch_size, 
                self.total_samples - (batch_id * self.batch_size)
            )
            
            jobs.append(
                delayed(self.worker)(
                    batch_id, 
                    current_batch_size, 
                    self.grid_params, 
                    self.sim_params, 
                    self.output_dir
                )
            )

        results = Parallel(n_jobs=self.n_jobs, verbose=5)(jobs)
        return results