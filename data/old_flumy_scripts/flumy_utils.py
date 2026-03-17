import os
import h5py
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

class FlumyDataManager:
    """
    Manages the creation of HDF5 files for 3D geological data
    """
    def __init__(self, output_dir : str="datasets/training"):
        """
        Args:
            output_dit (str): sets output directory for training, validation and testing data. defaults is "datasets/training"
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize_h5_file(self, batch_id, n_samples, nx, ny, nz):
        """
        Pre-allocates an HDF5 file with gzip compression for a specific batch.
        
        Args:
            batch_id (int): Unique identifier for this batch (used for naming).
            n_samples (int): Number of realizations this file will hold.
            nx (int): Grid size in the X direction.
            ny (int): Grid size in the Y direction.
            nz (int): Grid size in the Z (depth) direction.
            
        Returns:
            str: The absolute or relative path to the created HDF5 file.
        """
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
        """
        Writes a single 3D realization into a specific index slot of an existing HDF5 file.
        
        Args:
            file_path (str): Path to the target HDF5 file.
            index (int): The slot index (0 to n_samples-1) to save the data into.
            fac (np.ndarray): 3D array representing facies.
            grain (np.ndarray): 3D array representing grain size.
            age (np.ndarray): 3D array representing geological age.
        """
        with h5py.File(file_path, 'a') as f:
            f['facies'][index, ...] = fac.astype(np.uint8)
            f['grain'][index, ...] = grain.astype(np.float32)
            f['age'][index, ...] = age.astype(np.int32)

class BatchSimulator:
    """
    Orchestrates the parallel execution of simulation tasks across multiple CPU cores.
    """
    def __init__(self, worker_function, total_samples, batch_size, grid_params, sim_params, output_dir, n_jobs=-1):
        """
        Args:
            worker_function (Callable): The function executed by each parallel worker.
            total_samples (int): The total number of simulations to generate.
            batch_size (int): The number of simulations to group into a single HDF5 file.
            grid_params (dict): Dimensions and resolution of the 3D grid.
            sim_params (dict): Physical simulation parameters (seed, depths, etc.).
            output_dir (str): Destination directory for the output data.
            n_jobs (int): Number of CPU cores to utilize. Defaults to -1 (all available).
        """
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
        Calculates the required number of batches and dispatches jobs to the parallel backend.
        
        Returns:
            List[str]: A list of file paths to the generated HDF5 batch files.
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