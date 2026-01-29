from dataset import HashtagDataset
from similarity import relevance_score
import json

data = HashtagDataset("data/hashtags/processed.csv")

scores, labels = [], []

for i in range(len(data)):
    caption, hashtags, label = data[i]
    score = relevance_score(caption, hashtags)
    scores.append(score)
    labels.append(label)

# Simple threshold selection
threshold = sum(scores) / len(scores)

with open("training/hashtag_training/thresholds.json", "w") as f:
    json.dump({"threshold": round(threshold, 3)}, f)

print("Optimal threshold:", threshold)

