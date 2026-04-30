"""
Custom dataloader for facies classification. Reads a directory of .npz files containing 4D facies data. 
Option to use one-hot encoding to convert facies labels into channels.

Default mapping: 
    FA1: Fluvial and distributary channel deposits -> fac 0,1,2 & 3
    FA2: Crevasse splays and levee deposits -> fac 4,5,6 & 7
    FA3: Overbank deposits and paleosols -> fac 8,9 -> rest does not occur often
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    def __init__(self, root, facies_mapping=None, save_mapping_dir=None, use_one_hot=True, **kwargs):
        self.dir_path = Path(root)
        self.use_one_hot = use_one_hot
        
        # Grab all .npz files in the specified directory
        self.files = sorted(list(self.dir_path.glob("*.npz")))
        self.length = len(self.files)
        
        if self.length == 0:
            raise FileNotFoundError(f"No .npz files were found in the directory: {self.dir_path}")
        
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
        # Load the specific .npz file for this iteration
        npz_path = self.files[idx]
        
        # Open the npz archive
        with np.load(npz_path) as npz_file:
            # We dynamically grab the first array in the .npz file, 
            # in case the saving script used different keys (like 'arr_0' or 'facies')
            key = list(npz_file.keys())[0]
            data = npz_file[key]
        
        mapped_data = self.mapping[data]
        tensor_data = torch.from_numpy(mapped_data)
        
        if self.use_one_hot:
            # .permute changes shape from (D, H, W, Channels) to (Channels, D, H, W)
            processed_data = F.one_hot(tensor_data, num_classes=3).permute(3, 0, 1, 2).float()
            
            # SCALE TO [-1, 1] TO MATCH GENERATOR nn.Tanh()
            processed_data = (processed_data * 2.0) - 1.0
        else:
            # Add a channel dimension for the single index value: (1, D, H, W)
            processed_data = tensor_data.unsqueeze(0).float()
            
            # SCALE TO [-1, 0, 1] TO MATCH GENERATOR nn.Tanh() bounds
            processed_data = processed_data - 1.0

        return {'data': processed_data}