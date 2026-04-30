import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    def __init__(self, h5_path, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, **kwargs):
        self.h5_path = h5_path
        self.use_one_hot = use_one_hot
        
        # Open briefly just to get the total number of samples
        with h5py.File(self.h5_path, 'r') as h5f:
            if 'facies' not in h5f:
                raise KeyError(f"Dataset 'facies' not found in {self.h5_path}")
            self.length = h5f['facies'].shape[0]
        
        # Define mappings   
        if facies_mapping is None:
            self.mapping = np.zeros(13, dtype=np.int64)
            self.mapping[1:4] = 0   
            self.mapping[4:8] = 1   
            self.mapping[8:13] = 2  
            
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
        # Open file inside __getitem__ to ensure thread safety with PyTorch workers
        with h5py.File(self.h5_path, 'r') as h5f:
            data = h5f['facies'][idx]
        
        mapped_data = self.mapping[data]
        tensor_data = torch.from_numpy(mapped_data)
        
        if self.use_one_hot:
            # .permute changes shape from (D, H, W, Channels) to (Channels, D, H, W)
            processed_data = F.one_hot(tensor_data, num_classes=3).permute(3, 0, 1, 2).float()
            # SCALE TO [-1, 1] TO MATCH GENERATOR nn.Tanh()
            processed_data = (processed_data * 2.0) - 1.0
        else:
            processed_data = tensor_data.unsqueeze(0).float()
            processed_data = processed_data - 1.0

        return {'data': processed_data}