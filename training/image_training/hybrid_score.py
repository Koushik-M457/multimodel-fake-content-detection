from watermark import watermark_score

def hybrid_image_score(resnet_score, image_path,
                       resnet_weight=0.8,
                       watermark_weight=0.2,
                       threshold=0.6):
    """
    Combines ResNet-50 and watermark detection scores
    """

    # Step 1: Get watermark score
    wm_score = watermark_score(image_path)

    # Step 2: Weighted fusion
    final_score = (
        resnet_weight * resnet_score +
        watermark_weight * wm_score
    )

    # Step 3: Final decision
    verdict = "FAKE" if final_score >= threshold else "REAL"

    # Step 4: Return explainable output
    return {
        "resnet_score": round(resnet_score, 3),
        "watermark_score": round(wm_score, 3),
        "final_image_score": round(final_score, 3),
        "verdict": verdict
    }
