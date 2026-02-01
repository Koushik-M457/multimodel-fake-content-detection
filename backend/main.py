from fastapi import FastAPI, UploadFile, Form
from training.image_training.inference import load_image
from training.image_training.hybrid_score import hybrid_image_score
from training.text_training.inference import caption_fake_probability
from training.hashtag_training.inference import hashtag_relevance
from training.fusion_training.fusion import multimodal_fusion
import shutil
import os

app = FastAPI()


@app.post("/analyze")
async def analyze_post(
    image: UploadFile,
    caption: str = Form(...),
    hashtags: str = Form(...)
):
    # Save uploaded image
    image_path = f"temp_{image.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # IMAGE (Hybrid: ResNet + Watermark)
        image_result = hybrid_image_score(
            resnet_score=0.7,   # TODO: replace with real ResNet inference
            image_path=image_path
        )

        image_score = image_result["final_image_score"]
        watermark_score = image_result["watermark_score"]

        # TEXT
        text_score = caption_fake_probability(caption)

        # HASHTAGS
        hashtag_score = hashtag_relevance(caption, hashtags)["hashtag_score"]

        # FUSION (SAFE TWO-SIGNAL LOGIC)
        final_result = multimodal_fusion(
            image_score=image_score,
            watermark_score=watermark_score,
            text_score=text_score,
            hashtag_score=hashtag_score
        )

        return final_result

    finally:
        # Cleanup temp file
        if os.path.exists(image_path):
            os.remove(image_path)
