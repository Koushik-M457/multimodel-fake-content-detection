from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import RobertaForSequenceClassification
from dataset import CaptionDataset
from torch.utils.data import DataLoader

model = RobertaForSequenceClassification.from_pretrained("checkpoints/text")
model.eval()

test_data = CaptionDataset("data/text/test.csv")
loader = DataLoader(test_data, batch_size=16)

y_true, y_pred = [], []

with torch.no_grad():
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(preds.tolist())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))

