import io
from typing import List

# import numpy as np
import PIL
import tifffile
import torch
import torch.nn.functional as F
from eurosat_classification.data.label_map import label_map
from eurosat_classification.data.preprocessors import normalize_MS_img
from eurosat_classification.features.retrieve_model import get_model
from eurosat_classification.models.cnn import CNN
from eurosat_classification.features.integrated_gradients import load_rgb_ig, load_ms_ig, integrated_gradients, visualise_rgb, visualise_ms
from eurosat_classification.features.gradcam import gradcam, _scaled_rgb_colour
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from torchvision import transforms
import matplotlib.pyplot as plt

# Classes for prediction API
class ClassConfidence(BaseModel):
    class_pred: str
    confidence: float


class ClassPredictions(BaseModel):
    predictions: List[ClassConfidence]
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {"class_pred": "PermanentCrop", "confidence": 0.971},
                    {
                        "class_pred": "HerbaceousVegetation",
                        "confidence": 0.028,
                    },
                    {"class_pred": "Highway", "confidence": 0.001},
                    {"class_pred": "AnnualCrop", "confidence": 0},
                    {"class_pred": "Forest", "confidence": 0},
                    {"class_pred": "Industrial", "confidence": 0},
                    {"class_pred": "Pasture", "confidence": 0},
                    {"class_pred": "Residential", "confidence": 0},
                    {"class_pred": "River", "confidence": 0},
                    {"class_pred": "SeaLake", "confidence": 0},
                ]
            }
        }
    }


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
Expected input is a three-channel RGB satellite image, preferably 64 by 64
pixels, although other formats are supported.

## Model Usage - MS
This model is trained on 64x64 multispectral images of Sentinel-2 satellite
data. Expected input is a thirteen-channel multispectral satellite image.

    """,
    version="alpha",
)

# Load models
model_rgb = get_model("models/model1.pkl")
model_ms = get_model("models/model2.pkl")

# Functions for prediction API
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
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = normalize_MS_img(image).unsqueeze(0)
    return image.to(torch.float32)


def model_predict(model: CNN, image: torch.Tensor) -> ClassPredictions:
    """Classifies a given image using a given model."""
    confs = model(image).detach()[0]
    confs = F.softmax(confs, dim=0)
    class_confs = [
        ClassConfidence(
            class_pred=label_map[i], confidence=round(float(conf), 3)
        )
        for i, conf in enumerate(confs)
    ]
    class_confs = sorted(class_confs, key=lambda x: x.confidence, reverse=True)
    return ClassPredictions(predictions=class_confs)


# Functions for XAI API
def class_to_explain(preprocessed_img: torch.Tensor, image_type: str) -> int:
    with torch.no_grad():
        if image_type == "RGB":
            predicted_class = int(model_rgb(preprocessed_img.unsqueeze(0)).argmax(1).item())
    
        elif image_type == "MS":
            predicted_class = int(model_ms(preprocessed_img.unsqueeze(0)).argmax(1).item())

    return predicted_class
    

def ig_explain(raw, preprocessed, baseline, predicted_class, target_class, n_steps, image_type):
    if image_type == "RGB":
        attrs = integrated_gradients(model_rgb.eval(), preprocessed, baseline, target_class, n_steps)
        return visualise_rgb(raw, attrs, predicted_class, target_class, output_path=None)
    elif image_type == "MS":
        attrs = integrated_gradients(model_ms.eval(), preprocessed, baseline, target_class, n_steps)
        return visualise_ms(raw, attrs, predicted_class, target_class, output_path=None)

def gradcam_explain(img_tensor, img_array, predicted_class, target_class, image_type):
    if image_type == "RGB":
        gradcam_visualization = gradcam(model_rgb, img_tensor, img_array, target_class)
    elif image_type == "MS":
        gradcam_visualization = gradcam(model_ms, img_tensor, img_array, target_class)
    
    fig, ax = plt.subplots()
    ax.imshow(gradcam_visualization)
    ax.set_title(f"GradCAM | Predicted: {label_map[predicted_class]} | Explaining: {label_map[target_class]}")
    ax.axis("off")
    plt.show()

# API endpoints
@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url="/docs")

# Predict for RGB model (i.e. baseline)
@app.post(
    "/predict_rgb",
    summary="Predict class of a three-channel RGB satellite image.",
    description="Image classifier endpoint to classify three-channel RGB "
    "satellite images. Requests should be of the format multipart/form-data, "
    "and include an image sent using the applicable 'image' field. The given "
    "image should be a three-channel RGB satellite image using a 10m ground "
    "sampling distance. That is, the distance between the center of two "
    "consecutive pixels is 10m when measured on the ground. Returns class "
    "confidences for each of the ten supported classes, ranked by confidence "
    "score, in json format.",
    response_model=ClassPredictions,
    response_description="""Returns model confidences for the following
        classes, ranked by confidence score:\n
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

    return model_predict(model_rgb, tensor_image)

# Predict for MS model
@app.post(
    "/predict_ms",
    summary="Predict class of a multispectral (13-band) satellite image.",
    description="Image classifier endpoint to classify multispectral "
    "(13-band) satellite images. Requests should be of the format "
    "multipart/form-data, and include an image sent using the "
    "applicable 'image' field. The given image should be a "
    "thirteen-channel multispectral satellite image using a 10m ground "
    "sampling distance. That is, the distance between the center of "
    "two consecutive pixels is 10m when measured on the ground. "
    "Returns class confidences for each of the ten supported "
    "classes, ranked by confidence score, in json format.",
    response_model=ClassPredictions,
    response_description="""Returns model confidences for the following
        classes, ranked by confidence score:\n
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

    return model_predict(model_ms, tensor_image)

# Explainability for RGB (i.e. baseline) model - GradCAM & Integrated Gradients
@app.post(
    "/explain_rgb",
    summary="Explainability for RGB model.",
    description="description",
    response_description="returns gradcam heatmap for specified target class" \
    "of input, or 4x4 images with integrated gradients expalantaopns",
)
async def explain_rgb(image: UploadFile, target_class: int | None = None, n_steps: int = 50):
    try:
        # Integrated gradients
        img, baseline = load_rgb_ig(image.file)
        raw = preprocessed = img

        # GradCam
        tensor_image = process_image(image, "RGB")
        array_image = tensor_image.permute(1, 2, 0).numpy() 
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    
    predicted_class = class_to_explain(preprocessed, "RGB")
    if target_class is None:
        target_class = predicted_class

    return ig_explain(raw, preprocessed, baseline, predicted_class, target_class, n_steps, image_type="RGB"), gradcam_explain(tensor_image, array_image, predicted_class, target_class, image_type="RGB")



# Explainability for MS model - GradCAM & Integrated Gradients
@app.post(
    "/explain_ms",
    summary="Explainability for MS model.",
    description="description",
    response_description="returns gradcam heatmap for specified target class" \
    "of input, or 4x4 images with integrated gradients expalantaopns",
)
async def explain_ms(image: UploadFile, target_class: int | None = None, n_steps: int = 50):
    try:
        # Integrated gradients
        raw, preprocessed, baseline = load_ms_ig(image)

        # GradCAM
        tensor_image = process_image(image, "MS")
        array_image = _scaled_rgb_colour(tensor_image)
    except tifffile.tifffile.TiffFileError:
        raise HTTPException(
            status_code=415,
            detail="Invalid image extension specified. "
            "Multispectral image prediction accepts "
            "only TIF files.",
        )
    
    predicted_class = class_to_explain(preprocessed, "MS")
    if target_class is None:
        target_class = predicted_class

    return ig_explain(raw, preprocessed, baseline, predicted_class, target_class, n_steps, image_type="MS"), gradcam_explain(tensor_image, array_image, predicted_class, target_class, image_type="MS")
