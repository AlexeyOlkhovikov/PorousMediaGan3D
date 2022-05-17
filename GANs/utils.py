import torch
from torch.utils.data import Dataset
import numpy as np


class GanDataset(Dataset):
    def __init__(self, samples, transforms=None):
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx, :, :, :]
        img = img / 255.

        slice = img[0, :, :]
        slice = np.expand_dims(slice, 2)

        if self.transforms is not None:
            img = self.transforms(img).to(dtype=torch.float)
            slice = self.transforms(slice).to(dtype=torch.float)

        img = torch.unsqueeze(img, 0)

        return (img, slice)