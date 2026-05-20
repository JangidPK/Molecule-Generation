import torch
from torch.utils.data import Dataset


class SmilesDataset(Dataset):

    def __init__(self, sequences):

        self.X = [seq[:-1] for seq in sequences]
        self.Y = [seq[1:] for seq in sequences]


    def __len__(self):

        return len(self.X)


    def __getitem__(self, idx):

        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.Y[idx], dtype=torch.long)
        )