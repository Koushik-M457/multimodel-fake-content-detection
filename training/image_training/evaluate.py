import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import get_resnet50
from inference import load_image
from hybrid_score import hybrid_image_score
from resnet_inference import resnet_fake_probability

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = get_resnet50()
model.load_state_dict(torch.load("checkpoints/image/resnet50_best.pth"))
model.to(device)
model.eval()

# Load test dataset
test_dataset = ImageFolder("data/image/processed/test")
y_true, y_pred, y_scores = [], [], []

for img_path, label in test_dataset.samples:
    image_tensor = load_image(img_path)

    # ResNet score
    resnet_score = resnet_fake_probability(model, image_tensor, device)

    # Hybrid score
    result = hybrid_image_score(resnet_score, img_path)

    final_score = result["final_image_score"]
    prediction = 1 if result["verdict"] == "FAKE" else 0

    y_true.append(label)
    y_pred.append(prediction)
    y_scores.append(final_score)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_scores)

print("📊 IMAGE MODEL EVALUATION")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
