def multimodal_fusion(
    image_score,
    text_score,
    hashtag_score,
    image_weight=0.4,
    text_weight=0.4,
    hashtag_weight=0.2,
    threshold=0.6
):
    """
    All scores must be normalized between 0 and 1
    Higher score => more likely FAKE
    """

    final_score = (
        image_weight * image_score +
        text_weight * text_score +
        hashtag_weight * hashtag_score
    )

    verdict = "FAKE" if final_score >= threshold else "REAL"

    return {
        "image_score": round(image_score, 3),
        "text_score": round(text_score, 3),
        "hashtag_score": round(hashtag_score, 3),
        "final_score": round(final_score, 3),
        "verdict": verdict
    }

