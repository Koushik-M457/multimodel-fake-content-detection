from fastapi import FastAPI, UploadFile, Form
from training.image_training.inference import load_image
from training.image_training.hybrid_score import hybrid_image_score
from training.text_training.inference import caption_fake_probability
from training.hashtag_training.inference import hashtag_relevance
from training.fusion_training.fusion import multimodal_fusion
import torch
import shutil

app = FastAPI()

@app.post("/analyze")
async def analyze_post(
    image: UploadFile,
    caption: str = Form(...),
    hashtags: str = Form(...)
):
    image_path = f"temp_{image.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # IMAGE
    image_tensor = load_image(image_path)
    image_score = hybrid_image_score(
        resnet_score=0.7,  # replace with real inference
        image_path=image_path
    )["final_image_score"]

    # TEXT
    text_score = caption_fake_probability(caption)

    # HASHTAGS
    hashtag_score = hashtag_relevance(caption, hashtags)["hashtag_score"]

    # FUSION
    final_result = multimodal_fusion(
        image_score=image_score,
        text_score=text_score,
        hashtag_score=hashtag_score
    )

    return final_result

