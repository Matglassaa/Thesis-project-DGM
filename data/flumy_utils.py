# import os
# import numpy as np

# class FlumyDataManager:
#     """
#     A utility class for managing, saving, and validating Flumy simulation data.
    
#     This manager handles the conversion of raw simulation outputs into 
#     high-accuracy compressed storage and provides integrity verification
#     to ensure data consistency for machine learning datasets.
#     """
    
#     def __init__(self, output_dir="dataset"):
#         """
#         Initializes the Data Manager.

#         Args:
#             output_dir (str): The directory where realizations will be stored.
#                               Created automatically if it doesn't exist.
#         """
#         self.output_dir = output_dir
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)


#     def check_integrity(self, filename, original_data=None, verbose=True):
#         """
#         Validates the integrity and accuracy of a stored .npz file.

#         Args:
#             filename (str): Name of the file within the output directory.
#             original_data (dict, optional): A dictionary containing 'facies', 'grain', 
#                                             and 'age' arrays to compare against disk.
#             verbose (bool): If True, prints a detailed report to console.

#         Returns:
#             bool: True if the file exists, contains data, and matches original_data.
#         """
#         file_path = os.path.join(self.output_dir, filename)
#         if not os.path.exists(file_path):
#             if verbose: print(f"Error: {file_path} not found.")
#             return False

#         data = np.load(file_path)
#         is_valid = True
#         report = [f"--- Integrity Report: {filename} ---"]

#         # Check for zeros and types
#         for key in ['facies', 'grain', 'age']:
#             arr = data[key]
#             if np.all(arr == 0):
#                 report.append(f"  !! WARNING: {key} is all zeros!")
#                 is_valid = False

        
#         # Comparison logic
#         if original_data is not None:
#             matches = True
#             if not np.array_equal(data['facies'], original_data['facies'].astype(np.uint8)):
#                 matches = False
#             if not np.allclose(data['grain'], original_data['grain'].astype(np.float32), atol=1e-7):
#                 matches = False
#             if not np.array_equal(data['age'], original_data['age'].astype(np.int32)):
#                 matches = False
            
#             report.append("  SUCCESS: 100% Match" if matches else "  !! ERROR: Data Mismatch")
#             is_valid = is_valid and matches

#         if verbose:
#             print("\n".join(report))
#         return is_valid

#     def save_realization(self, sim_id, fac, grain, age):
#         """
#         Saves a 3D simulation block to a compressed .npz file with type preservation.

#         Args:
#             sim_id (int): Unique identifier for the realization.
#             fac (np.ndarray): Facies identifiers (converted to uint8).
#             grain (np.ndarray): Grain size values (preserved as float32).
#             age (np.ndarray): Iteration/Time values (preserved as int32).

#         Returns:
#             str: The full path to the saved file.
#         """
#         file_path = os.path.join(self.output_dir, f"realization_{sim_id:05d}.npz")
#         np.savez_compressed(
#             file_path,
#             facies=fac.astype(np.uint8),
#             grain=grain.astype(np.float32),
#             age=age.astype(np.int32)
#         )
#         return file_path

# # This only runs if you run 'python flumy_utils.py' directly
# if __name__ == "__main__":
#     print("Running internal module test...")
#     manager = FlumyDataManager(output_dir="test_dir")

import os
import numpy as np
import h5py

class FlumyDataManager:
    """
    Utility for managing Flumy simulation data in HDF5 format.
    Optimized for high-throughput generation on multi-core servers.
    """
    
    def __init__(self, output_dir="datasets/training"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize_h5_file(self, batch_id, n_samples, nx, ny, nz):
        """
        Pre-allocates an HDF5 file for a specific batch/core.
        """
        file_path = os.path.join(self.output_dir, f"batch_core_{batch_id:03d}.h5")
        
        with h5py.File(file_path, 'w') as f:
            # Create datasets with compression for 3D regular blocks
            # Datatypes match Flumy outputs: facies (uint8), grain (float32), age (int32)
            f.create_dataset("facies", (n_samples, nx, ny, nz), dtype='u1', compression="gzip")
            f.create_dataset("grain", (n_samples, nx, ny, nz), dtype='f4', compression="gzip")
            f.create_dataset("age", (n_samples, nx, ny, nz), dtype='i4', compression="gzip")
        
        return file_path

    def save_to_batch(self, file_path, index, fac, grain, age):
        """
        Saves a single realization into a specific index of a batch file.
        """
        with h5py.File(file_path, 'a') as f:
            f['facies'][index, ...] = fac.astype(np.uint8)
            f['grain'][index, ...] = grain.astype(np.float32)
            f['age'][index, ...] = age.astype(np.int32)

    def check_integrity(self, file_path, index, original_data=None, verbose=True):
        """
        Validates a specific sample within an HDF5 batch file.
        """
        if not os.path.exists(file_path):
            if verbose: print(f"Error: {file_path} not found.")
            return False

        with h5py.File(file_path, 'r') as f:
            fac = f['facies'][index]
            grain = f['grain'][index]
            age = f['age'][index]

            is_valid = True
            report = [f"--- Integrity Report: Batch {file_path} Sample {index} ---"]

            # Standard checks (aligned with Flumy Lithofacies/Grain Size specs)
            for key, arr in zip(['facies', 'grain', 'age'], [fac, grain, age]):
                if np.all(arr == 0) and key != 'grain': # Grain size can be 0.125-1.0
                    report.append(f"  !! WARNING: {key} is all zeros!")
                    is_valid = False

            if original_data is not None:
                matches = (np.array_equal(fac, original_data['facies'].astype(np.uint8)) and
                           np.allclose(grain, original_data['grain'].astype(np.float32), atol=1e-7) and
                           np.array_equal(age, original_data['age'].astype(np.int32)))
                
                report.append("  SUCCESS: 100% Match" if matches else "  !! ERROR: Data Mismatch")
                is_valid = is_valid and matches

            if verbose:
                print("\n".join(report))
            return is_valid