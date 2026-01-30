import json
from .similarity import relevance_score


with open("training/hashtag_training/thresholds.json") as f:
    THRESHOLD = json.load(f)["threshold"]

def hashtag_relevance(caption, hashtags):
    score = relevance_score(caption, hashtags)
    verdict = "RELEVANT" if score >= THRESHOLD else "IRRELEVANT"

    return {
        "hashtag_score": round(score, 3),
        "verdict": verdict
    }

