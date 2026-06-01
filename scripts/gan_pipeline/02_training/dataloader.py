import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    def __init__(self, h5_path=None, num_samples=None, nz=32, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, one_hot_all=False, preload_ram=True, **kwargs):        
        self.h5_path = h5_path
        self.nz = nz
        self.use_one_hot = use_one_hot
        self.one_hot_all = one_hot_all
        self.preload_ram = preload_ram
        
        self.num_classes = 10 if self.one_hot_all else 3
        
        # --- MODIFICATION: Bypass H5 loading if h5_path is None ---
        if self.h5_path is not None:
            with h5py.File(self.h5_path, 'r') as h5f:               
                if 'facies' not in h5f:
                    raise KeyError(f"Dataset 'facies' not found in {self.h5_path}")
                
                total_length = h5f['facies'].shape[0]
                self.length = min(num_samples, total_length) if num_samples is not None else total_length
                
                if self.preload_ram:
                    self.data_cache = h5f['facies'][:self.length, -nz:, :, :].astype(np.uint8) 
        else:
            # Debugging mode active: no H5 file needed
            self.length = 1
            self.preload_ram = False
        
        # --- Mapping Logic (Unchanged) ---
        if facies_mapping is None:
            if self.one_hot_all:
                self.mapping = np.arange(13, dtype=np.uint8) 
                self.mapping[1:10] = np.arange(9) 
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # --- MODIFICATION: Inspect .npy files directly ---
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
            # Determine correct permute order based on input shape dynamically
            dims = len(tensor_data.shape) + 1 # +1 for the new one-hot class dimension
            permute_order = (dims - 1,) + tuple(range(dims - 1))
            
            processed_data = F.one_hot(tensor_data, num_classes=self.num_classes).permute(*permute_order).float()
            #processed_data = (processed_data * 2.0) - 1.0
        else:
            processed_data = tensor_data.unsqueeze(0).float()
            #processed_data = processed_data - 1.0

        if is_inspecting:
            print(f"3. FINAL TENSOR| Shape: {processed_data.shape} | Unique Vals: {torch.unique(processed_data).tolist()}\n")

        return {'data': processed_data}


if __name__ == "__main__":
    TEST_NPY_PATH = "data/test_outputs_lower_plain_delta_nz_32/sample_1_facies.npy"
    
    # 1. Create a quick dummy .npy file if you don't have one ready
    if not os.path.exists(TEST_NPY_PATH):
        # Creates a 32x64x64 array containing only raw facies values 1, 5, and 12
        dummy_data = np.random.choice([1, 5, 12], size=(32, 64, 64)).astype(np.uint8)
        np.save(TEST_NPY_PATH, dummy_data)

    # 2. Initialize the dataset with NO h5_path (h5_path=None)
    # This sets up the mapping logic without touching disk I/O
    dataset = FaciesDataset(
        h5_path=None, 
        use_one_hot=True, 
        one_hot_all=True
    )
    
    # 3. Feed it the .npy path directly to trigger the printouts
    _ = dataset[TEST_NPY_PATH]