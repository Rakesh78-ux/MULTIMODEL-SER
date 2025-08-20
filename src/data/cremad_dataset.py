import torch
from torch.utils.data import Dataset
import pandas as pd

class CremadDataset(Dataset):
    def __init__(self, pkl_path, split="train", train_ratio=0.8):
        df = pd.read_pickle(pkl_path)

        # train/test split
        train_size = int(len(df) * train_ratio)
        if split == "train":
            self.df = df.iloc[:train_size]
        else:
            self.df = df.iloc[train_size:]

        # labels mapping
        self.labels = sorted(self.df["label"].unique())
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row["features"], dtype=torch.float32)
        label = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return features, label
