from fastapi import FastAPI, UploadFile, Form
from training.image_training.inference import (
    load_image,
    load_resnet_model,
    resnet_fake_probability
)
from training.image_training.hybrid_score import hybrid_image_score
from training.text_training.inference import caption_fake_probability
from training.hashtag_training.inference import hashtag_relevance
from training.fusion_training.fusion import multimodal_fusion
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model once
model = load_resnet_model()


@app.post("/analyze")
async def analyze_post(
    image: UploadFile,
    caption: str = Form(...),
    hashtags: str = Form(...)
):
    image_path = f"temp_{image.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # IMAGE
        image_tensor = load_image(image_path)
        resnet_score = resnet_fake_probability(model, image_tensor)

        image_result = hybrid_image_score(
            resnet_score=resnet_score,
            image_path=image_path
        )

        image_score = image_result["final_image_score"]
        watermark_score = image_result["watermark_score"]

        # TEXT
        text_score = caption_fake_probability(caption)

        # HASHTAGS
        hashtag_score = hashtag_relevance(caption, hashtags)["hashtag_score"]

        # FUSION
        final_result = multimodal_fusion(
            image_score=image_score,
            watermark_score=watermark_score,
            text_score=text_score,
            hashtag_score=hashtag_score
        )

        return final_result

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)