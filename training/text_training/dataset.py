import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class CaptionDataset(Dataset):
    def __init__(self, csv_file, max_len=128):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx]["caption"]
        label = int(self.data.iloc[idx]["label"])

        encoding = tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

