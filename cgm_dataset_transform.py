import numpy as np
import torch

# --- Transform class ---
class GlucoseTransform:
    def __init__(self, add_range=(0,0), alpha_range=(1.0,1.0), noise_range=(0,0)):
        """
        add_range: tuple (low, high) range of constant to add
        alpha_range: tuple (low, high) range for random scaling multiplier
        noise_range: tuple (low, high) for random uniform noise
        """
        self.add_range = add_range
        self.alpha_range = alpha_range
        self.noise_range = noise_range

    def __call__(self, x):
        """
        x: torch.Tensor of shape (1, T) or (N, 1, T)
        returns transformed tensor with same shape
        """
        # pick a random constant to add
        c = np.random.uniform(*self.add_range)
        # random scale factor
        a = np.random.uniform(*self.alpha_range)
        # noise sampled per element
        noise = np.random.uniform(self.noise_range[0], self.noise_range[1], size=x.shape)
        
        # Apply transforms
        x = x + c
        x = x * a
        x = x + torch.from_numpy(noise).type_as(x)
        return x

import pandas as pd
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 

class CGMDataset(Dataset):
    def __init__(self, df, seq_col='sequence', label_col='label',
                 offset=0, mode=0, transform=None):
        sequences = np.stack(df[seq_col].values).astype(np.float32)
        labels    = df[label_col].values.astype(np.float32)
        self.mode = mode
        self.mean = 140
        self.std  = 120
        self.offset = offset
        self.transform = transform

        sequences = (sequences - self.mean) / self.std

        self.X = torch.from_numpy(sequences).unsqueeze(1)
        self.y = torch.from_numpy(labels).unsqueeze(1)

    def __len__(self):
        return len(self.X)*7 if self.mode == 0 else len(self.X)

    def __getitem__(self, idx):
        if self.mode == 0:
            patient_id = idx // 7
            day = idx - 7*patient_id
            x = self.X[patient_id,:,day*288:day*288+288]
            y = self.y[patient_id]
        elif self.mode == 1:
            x = self.X[idx]
            y = self.y[idx]
        elif self.mode == 2:
            x = self.X[idx,:,self.offset*288:self.offset*288+288]
            y = self.y[idx]

        # apply transform if provided
        if self.transform is not None:
            x = self.transform(x)

        return x, y