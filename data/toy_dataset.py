from torch.utils.data import Dataset
import torch

class ToyDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.X[idx]
