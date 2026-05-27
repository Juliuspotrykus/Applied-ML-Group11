import io
from typing import List

# import numpy as np
import PIL
import tifffile
import torch
import torch.nn.functional as F
from eurosat_classification.data.label_map import label_map
from eurosat_classification.data.preprocessors import normalize_MS_img
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from torchvision import transforms


class ClassConfidence(BaseModel):
    class_pred: str = "AnnualCrop"
    confidence: float = 0.1


class ClassPredictions(BaseModel):
    predictions: List[ClassConfidence]


# Define API and description
app = FastAPI(
    title="Satellite Image Classifier",
    summary="""An API endpoint to classify satellite images into one of ten
            classes using a CNN. Trained using the EuroSAT dataset.""",
    description="""
# An API endpoint to access CNN classifiers trained on the EuroSAT dataset.
Two models are accessible; one trained on RGB images (3-dimensional),
and one trained on multispectral images (13-dimensional).

## Model Usage - RGB
This model is trained on 64x64 RGB images of Sentinel-2 satellite data.

## Model Usage - MS
This model is trained on 64x64 multispectral images of Sentinel-2 satellite
data.

    """,
    version="alpha",
)


# Load RGB and MS models here. We have the full model stored,
# so we have to make use of weights_only=False.
model_rgb = torch.load("models/model1.pkl", weights_only=False)
model_ms = torch.load("models/model2.pkl", weights_only=False)


def process_image(file: UploadFile, image_type: str) -> torch.Tensor:
    """Processes an image given by a user."""
    if image_type == "RGB":
        # Resize, normalize, and add batch dimension
        image = Image.open(file.file).convert("RGB").resize((64, 64))
        image = transforms.ToTensor()(image).unsqueeze(0)
    elif image_type == "MS":
        # read TIFF from bytes, direct approach
        # using tiff.imread(file.file) crashes.
        tif_bytes = file.file.read()
        image = tifffile.imread(io.BytesIO(tif_bytes))
        image = torch.from_numpy(image).permute(2, 0, 1).resize((64, 64))
        image = normalize_MS_img(image).unsqueeze(0)
    return image.to(torch.float32)


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/predict_rgb",
    summary="Predict class of a three-channel RGB satellite image.",
    description="Image classifier endpoint. Add {'image': binary_image} "
    "to json body to send request. This image should be an RGB satellite "
    "image using a 10m ground sampling distance. That is, the distance "
    "between the center of two consecutive pixels is 10m when measured "
    "on the ground. Returns class confidences.",
    response_model=ClassPredictions,
    response_description="""Returns model confidences for the following
        classes:\n
        - AnnualCrop
        - Forest
        - HerbaceousVegetation
        - Highway
        - Industrial
        - Pasture
        - PermanentCrop
        - Residential
        - River
        - SeaLake""",
)
async def predict_rgb(image: UploadFile):
    # for final API version, create something
    # that can handle more than 10x10 meter images.
    try:
        tensor_image = process_image(image, "RGB")
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")

    confs = model_rgb(tensor_image).detach()[0]
    confs = F.softmax(confs, dim=0)
    class_confs = [
        ClassConfidence(class_pred=label_map[i], confidence=conf)
        for i, conf in enumerate(confs)
    ]
    return ClassPredictions(predictions=class_confs)


@app.post(
    "/predict_ms",
    summary="Predict class of a multispectral (13-band) satellite image.",
    description="Image classifier endpoint. Add {'image': binary_image} "
    "to json body to send request. This image should be a multispectral "
    "(13-band) satellite image using a 10m ground sampling distance. "
    "That is, the distance between the center of two consecutive pixels "
    "is 10m when measured on the ground. Returns class confidences.",
    response_model=ClassPredictions,
    response_description="""Returns model confidences for the following
        classes:\n
        - AnnualCrop
        - Forest
        - HerbaceousVegetation
        - Highway
        - Industrial
        - Pasture
        - PermanentCrop
        - Residential
        - River
        - SeaLake""",
)
async def predict_ms(image: UploadFile):
    try:
        tensor_image = process_image(image, "MS")
    except tifffile.tifffile.TiffFileError:
        raise HTTPException(
            status_code=415,
            detail="Invalid image extension specified. "
            "Multispectral image prediction accepts "
            "only TIF files.",
        )

    confs = model_ms(tensor_image).detach()[0]
    confs = F.softmax(confs, dim=0)
    class_confs = [
        ClassConfidence(class_pred=label_map[i], confidence=round(conf))
        for i, conf in enumerate(confs)
    ]

    return ClassPredictions(predictions=class_confs)
