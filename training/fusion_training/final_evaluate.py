import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

data = pd.read_csv("data/final_eval.csv")

# Fusion weights (best from tuning)
IMAGE_W = 0.4
TEXT_W = 0.4
HASH_W = 0.2
THRESHOLD = 0.6

y_true, y_pred, y_scores = [], [], []

for _, row in data.iterrows():
    final_score = (
        IMAGE_W * row["image_score"] +
        TEXT_W * row["text_score"] +
        HASH_W * row["hashtag_score"]
    )

    y_scores.append(final_score)
    y_pred.append(1 if final_score >= THRESHOLD else 0)
    y_true.append(row["label"])

# Metrics
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_scores))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

