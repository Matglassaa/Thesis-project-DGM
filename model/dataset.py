import torch
import numpy as np
from pathlib import Path

class FaciesDataset(Dataset):
    def __init__(self, root, transform=None):  # <-- Change root_dir to root
        self.root_dir = Path(root)             # <-- You can leave the rest exactly the same
        self.files = [f for f in os.listdir(root) if f.endswith('.npz')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.root_dir / self.files[idx]
        # Load the data and extract the 'facies' array
        data = np.load(filepath)['facies']
        
        # Add channel dimension -> (1, 64, 256, 256)
        data = np.expand_dims(data, axis=0) 
        
        # Convert to float tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)

        # Normalize from [1, 8] to [-1, 1]
        tensor_data = 2.0 * ((tensor_data - 1.0) / (8.0 - 1.0)) - 1.0

        if self.transform:
            tensor_data = self.transform(tensor_data)

        # --- THE FIX: Return as a dictionary ---
        return {'data': tensor_data}
