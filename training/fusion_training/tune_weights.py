import pandas as pd
from sklearn.metrics import f1_score

data = pd.read_csv("data/fusion_eval.csv")

best_f1 = 0
best_weights = None

threshold = 0.6

# Grid search (simple & explainable)
for iw in [0.2, 0.3, 0.4, 0.5]:
    for tw in [0.2, 0.3, 0.4, 0.5]:
        for hw in [0.1, 0.2, 0.3]:

            if abs(iw + tw + hw - 1.0) > 0.01:
                continue

            preds = []
            for _, row in data.iterrows():
                final_score = (
                    iw * row["image_score"] +
                    tw * row["text_score"] +
                    hw * row["hashtag_score"]
                )
                preds.append(1 if final_score >= threshold else 0)

            f1 = f1_score(data["label"], preds)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = (iw, tw, hw)

print(" Best F1-score:", round(best_f1, 4))
print(" Best weights:")
print(f"Image: {best_weights[0]}")
print(f"Text: {best_weights[1]}")
print(f"Hashtag: {best_weights[2]}")

