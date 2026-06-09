import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    def __init__(self, h5_path=None, num_samples=None, nz=32, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, one_hot_all=False, preload_ram=True, unique_raw_facies=None, **kwargs):        
        self.h5_path = h5_path
        self.nz = nz
        self.use_one_hot = use_one_hot
        self.one_hot_all = one_hot_all
        self.preload_ram = preload_ram
        
        # 1. Load h5py file
        if self.h5_path is not None:
            with h5py.File(self.h5_path, 'r') as h5f:               
                if 'facies' not in h5f:
                    raise KeyError(f"Dataset 'facies' not found in {self.h5_path}")
                
                total_length = h5f['facies'].shape[0]
                self.length = min(num_samples, total_length) if num_samples is not None else total_length
                
                if self.preload_ram:
                    self.data_cache = h5f['facies'][:self.length, -nz:, :, :].astype(np.uint8) 
        else:
            self.length = 1
            self.preload_ram = False
        
        # 2. Define facies mapping: Two options: either use one hot all or divert back to 3 classes (FA1, FA2 and FA3) 
        if facies_mapping is None:
            if self.one_hot_all:
                if unique_raw_facies is None:
                    raise ValueError("unique_raw_facies must be provided if one_hot_all is True")
                self.num_classes = len(unique_raw_facies)
                self.mapping = np.zeros(max(unique_raw_facies) + 1, dtype=np.int64)
                self.reverse_mapping = {}
                
                # Dynamically map whatever values exist to 0, 1, 2, 3...
                for new_idx, raw_val in enumerate(sorted(unique_raw_facies)):
                    self.mapping[raw_val] = new_idx
                    self.reverse_mapping[new_idx] = raw_val
            else:
                self.mapping = np.zeros(13, dtype=np.int64)
                self.mapping[1:4] = 0   
                self.mapping[4:8] = 1 
                self.mapping[8:13] = 2
                self.reverse_mapping = {0: 1, 1: 4, 2: 8}

        # 2.1. Custom facies mapping
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
            "representative_reverse_mapping": {str(k): v for k, v in self.reverse_mapping.items()}, # Stringify keys for clean JSON serializing
            "used_one_hot": self.use_one_hot,
            "num_classes": self.num_classes
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_inspecting = isinstance(idx, str) and idx.endswith('.npy')
        
        if is_inspecting:
            data = np.load(idx).astype(np.uint8)
            print(f"\n--- INSPECTING: {idx} ---")
            print(f"1. RAW DATA    | Shape: {data.shape} | Unique Vals: {np.unique(data)}")
        elif self.preload_ram:
            data = self.data_cache[idx]
        else:
            with h5py.File(self.h5_path, 'r') as h5f:
                data = h5f['facies'][idx].astype(np.uint8)
        
        mapped_data = self.mapping[data]
        
        if is_inspecting:
            print(f"2. MAPPED DATA | Shape: {mapped_data.shape} | Unique Vals: {np.unique(mapped_data)}")
            
        tensor_data = torch.from_numpy(mapped_data).long()
        
        if self.use_one_hot:
            dims = len(tensor_data.shape) + 1 
            permute_order = (dims - 1,) + tuple(range(dims - 1))
            
            processed_data = F.one_hot(tensor_data, num_classes=self.num_classes).permute(*permute_order).float()
            
            processed_data = (processed_data * 2.0) - 1.0
        else:
            processed_data = tensor_data.unsqueeze(0).float()
            processed_data = processed_data - 1.0

        if is_inspecting:
            print(f"3. FINAL TENSOR| Shape: {processed_data.shape} | Unique Vals: {torch.unique(processed_data).tolist()}\n")

        return {'data': processed_data}


# if __name__ == "__main__":
#     TEST_NPY_PATH = "data/test_outputs_lower_plain_delta_nz_32/sample_1_facies.npy"
    
#     # 1. Create a quick dummy .npy file if you don't have one ready
#     if not os.path.exists(TEST_NPY_PATH):
#         # Creates a 32x64x64 array containing only raw facies values 1, 5, and 12
#         dummy_data = np.random.choice([1, 5, 12], size=(32, 64, 64)).astype(np.uint8)
#         np.save(TEST_NPY_PATH, dummy_data)

#     # 2. Initialize the dataset with NO h5_path (h5_path=None)
#     # This sets up the mapping logic without touching disk I/O
#     dataset = FaciesDataset(
#         h5_path=None, 
#         use_one_hot=True, 
#         one_hot_all=True
#     )
    
#     # 3. Feed it the .npy path directly to trigger the printouts
#     _ = dataset[TEST_NPY_PATH]