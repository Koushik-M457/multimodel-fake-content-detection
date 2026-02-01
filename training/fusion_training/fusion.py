def multimodal_fusion(
    image_score,
    watermark_score,
    text_score,
    hashtag_score,
    image_weight=0.5,
    text_weight=0.3,
    hashtag_weight=0.2
):
    final_score = (
        image_weight * image_score +
        text_weight * text_score +
        hashtag_weight * hashtag_score
    )

    if final_score >= 0.65:
        verdict = "FAKE"
        confidence = "HIGH"
    elif final_score >= 0.45:
        verdict = "UNCERTAIN"
        confidence = "MEDIUM"
    else:
        verdict = "REAL"
        confidence = "HIGH"

    return {
        "image_score": round(image_score, 3),
        "watermark_score": round(watermark_score, 3),
        "text_score": round(text_score, 3),
        "hashtag_score": round(hashtag_score, 3),
        "final_score": round(final_score, 3),
        "verdict": verdict,
        "confidence": confidence
    }
