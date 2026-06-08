import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    """ 
    A PyTorch Dataset for loading and processing facies data from an HDF5 file.

    This dataset handles loading 3D or 2D facies arrays from an HDF5 file, mapping their raw integer values to defined classes, and applying preprocessing 
    such as one-hot encoding and scaling. It also supports preloading data into RAM to minimize disk I/O bottlenecks during training.

    Attributes:
        h5_path (str): Path to the HDF5 file containing the data.
        nz (int): Slicing parameter, typically determining the number of depth slices loaded.
        use_one_hot (bool): Indicates if the loaded data is converted to a one-hot representation.
        preload_ram (bool): Indicates if the dataset slices are preloaded into memory.
        length (int): Total number of samples along the primary axis of the dataset.
        data_cache (numpy.ndarray): In-memory cache of the data (if preload_ram is True).
        mapping (numpy.ndarray): Look-up array for fast forward mapping of facies values.
        reverse_mapping (dict): Dictionary mapping new values back to a representative raw value.
    """
    def __init__(self, h5_path, num_samples, nz=32, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, one_hot_all=False, preload_ram=True, **kwargs):        
        """
            Initializes the FaciesDataset.

            Args:
                h5_path (str): Path to the HDF5 file containing the 'facies' dataset.
                nz (int, optional): The number of elements to slice from the end of the first 
                    dimension (e.g., depth or sequence). Defaults to 64.
                facies_mapping (dict, optional): Dictionary defining a custom mapping from raw 
                    facies values to new class indices. If None, a default 3-class mapping is used. 
                    Defaults to None.
                save_mapping_dir (str, optional): Directory path where the mapping configuration 
                    should be saved as a JSON file. Defaults to None.
                use_one_hot (bool, optional): Whether to one-hot encode the processed data and scale 
                    it to [-1.0, 1.0]. Defaults to True.
                preload_ram (bool, optional): Whether to load the targeted slice of the dataset 
                    into RAM upon initialization. Defaults to True.
                **kwargs: Additional keyword arguments.

            Raises:
                KeyError: If the 'facies' key is not found within the specified HDF5 file.
        """
        self.h5_path = h5_path
        self.nz = nz
        self.use_one_hot = use_one_hot
        self.one_hot_all = one_hot_all # New parameter
        self.preload_ram = preload_ram
        
        # Determine number of classes for one-hot encoding
        self.num_classes = 9 if self.one_hot_all else 3
        
        with h5py.File(self.h5_path, 'r') as h5f:               
            if 'facies' not in h5f:
                raise KeyError(f"Dataset 'facies' not found in {self.h5_path}")
            
            total_length = h5f['facies'].shape[0]
            self.length = min(num_samples, total_length) if num_samples is not None else total_length
            
            if self.preload_ram:
                print(f"Loading {self.length} samples into RAM. Please wait...")
                # Best Practice: Cast to uint8 here to save massive RAM if raw values are just 0-12
                self.data_cache = h5f['facies'][:self.length, -nz:, :, :].astype(np.uint8) 
                num_loaded, depth, height, width = self.data_cache.shape
                print(f"Loading {num_loaded} samples into RAM complete!\n")
        
        # Define mappings   
        if facies_mapping is None:
            if self.one_hot_all:
                # Assuming raw facies values map directly to 0-8 for PyTorch one_hot requirements
                # If your raw data is 1-9, you must shift it to 0-8. Example below:
                self.mapping = np.arange(13, dtype=np.uint8) 
                self.mapping[1:10] = np.arange(9) # Uncomment if raw values are 1-9
                self.reverse_mapping = {i: i for i in range(9)}
            else:
                self.mapping = np.zeros(13, dtype=np.int64)
                self.mapping[1:4] = 0   
                self.mapping[4:8] = 1 
                self.mapping[9] = 0  
                self.mapping[[8, 10, 11, 12]] = 2
                self.reverse_mapping = {0: 1, 1: 4, 2: 8}
        else:
            max_val = max(facies_mapping.keys())
            self.mapping = np.zeros(max_val + 1, dtype=np.int64)
            self.reverse_mapping = {}
            for raw_val, new_val in facies_mapping.items():
                self.mapping[raw_val] = new_val
                if new_val not in self.reverse_mapping:
                    self.reverse_mapping[new_val] = raw_val

        if save_mapping_dir:
            self._save_mapping(save_mapping_dir)

    def _save_mapping(self, save_dir):
        """
        Saves the facies mapping configuration to a JSON file.

        Creates the directory if it does not exist and writes the forward mapping array,
        representative reverse mapping, and one-hot configuration to 'facies_mapping_config.json'.

        Args:
            save_dir (str): The directory where the configuration file will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "facies_mapping_config.json")
        config = {
            "forward_mapping_array": self.mapping.tolist(),
            "representative_reverse_mapping": self.reverse_mapping,
            "used_one_hot": self.use_one_hot
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves and processes a specific sample from the dataset.
        *DEBUG FEATURE*: Pass a string ending in '.npy' to inspect a specific file.
        """
        is_inspecting = isinstance(idx, str) and idx.endswith('.npy')
        
        if is_inspecting:
            data = np.load(idx).astype(np.uint8)
            print(f"--- INSPECTING: {idx} ---")
            print(f"1. RAW DATA    | Shape: {data.shape} | Unique Vals: {np.unique(data)}")
        elif self.preload_ram:
            data = self.data_cache[idx]
        else:
            with h5py.File(self.h5_path, 'r') as h5f:
                data = h5f['facies'][idx].astype(np.uint8)
        
        # --- 2. APPLY MAPPING ---
        mapped_data = self.mapping[data]
        
        if is_inspecting:
            print(f"2. MAPPED DATA | Shape: {mapped_data.shape} | Unique Vals: {np.unique(mapped_data)}")
            
        tensor_data = torch.from_numpy(mapped_data).long()
        
        # --- 3. FORMATTING (ONE-HOT & SCALING) ---
        if self.use_one_hot:
            processed_data = F.one_hot(tensor_data, num_classes=self.num_classes).permute(3, 0, 1, 2).float()
            processed_data = (processed_data * 2.0) - 1.0
        else:
            processed_data = tensor_data.unsqueeze(0).float()
            processed_data = processed_data - 1.0

        if is_inspecting:
            print(f"3. FINAL TENSOR| Shape: {processed_data.shape} | Unique Vals: {torch.unique(processed_data).tolist()}\n")

        return {'data': processed_data} 

if __name__ == "__main__":
    import os
    
    # ---------------------------------------------------------
    # 1. Define your paths here
    # ---------------------------------------------------------
    # The HDF5 file must exist so the class can initialize without error
    REAL_H5_PATH = "path/to/your/actual_dataset.h5" 
    TEST_NPY_PATH = "data/test_outputs_lower_plain_delta_nz_32/sample_1_facies.npy"
    
    # ---------------------------------------------------------
    # 2. Generate a dummy .npy file for testing (if needed)
    # ---------------------------------------------------------
    if not os.path.exists(TEST_NPY_PATH):
        print(f"Creating a dummy .npy file at {TEST_NPY_PATH}...")
        # Simulating a 3D crop (e.g., Depth x Height x Width)
        dummy_data = np.random.choice([1, 5, 10], size=(32, 64, 64)).astype(np.uint8)
        np.save(TEST_NPY_PATH, dummy_data)
        print("Dummy file created.\n")

    # ---------------------------------------------------------
    # 3. Run the Inspection
    # ---------------------------------------------------------
    try:
        # Initialize the dataset (requires valid H5 path to pass __init__ checks)
        print("Initializing FaciesDataset...")
        dataset = FaciesDataset(
            h5_path=REAL_H5_PATH, 
            num_samples=10,        # Keep small for quick init
            use_one_hot=True, 
            one_hot_all=False,
            preload_ram=False      # Turn off preload to save time during debugging
        )
        print("Initialization successful.\n")
        
        # Trigger the inspection by passing the .npy string instead of an integer
        inspection_result = dataset[TEST_NPY_PATH]
        
        # Verify the final output
        final_tensor = inspection_result['data']
        print(f"--- SUCCESS ---")
        print(f"Final output is a Tensor of type {final_tensor.dtype} and shape {final_tensor.shape}")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find the HDF5 file at '{REAL_H5_PATH}'.")
        print("Please update REAL_H5_PATH to point to your actual dataset file so the class can initialize.")
    except KeyError as e:
        print(f"ERROR: {e}")