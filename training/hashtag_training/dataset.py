import pandas as pd

class HashtagDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row["caption"], row["hashtags"], int(row["label"])

