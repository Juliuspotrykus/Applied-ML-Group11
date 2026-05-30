import io
from typing import List

import numpy as np
import PIL
import tifffile
import torch
import torch.nn.functional as F
from eurosat_classification.data.label_map import label_map, reverse_label_map
from eurosat_classification.data.preprocessors import normalize_MS_img
from eurosat_classification.features.retrieve_model import get_model
from eurosat_classification.models.cnn import CNN
from eurosat_classification.features.integrated_gradients import load_rgb_ig, load_ms_ig, integrated_gradients, visualise_rgb, visualise_ms
from eurosat_classification.features.gradcam import gradcam, _scaled_rgb_colour
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

## Classes for prediction API
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


## Define API app
app = FastAPI(
    title="Satellite Image Classifier",
    summary="""An API endpoint to classify satellite images into one of ten
            classes using a CNN. Trained using the EuroSAT dataset.""",
    description="""
# An API endpoint to access CNN classifiers trained on the EuroSAT dataset.
Two models are accessible; a baseline one trained on RGB images (3-dimensional),
and one trained on multispectral images (13-dimensional).

Ten classes into which satellite images can be classified and their index:
| Index | Class Name |
|-------|------|
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |

## Model Usage - RGB
This model is trained on 64x64 RGB images of Sentinel-2 satellite data.
Expected input is a three-channel RGB satellite image, preferably 64 by 64
pixels, although other formats are supported. The ground sampling distance
should be 10 meters.

## Model Usage - Multi-spectral (MS)
This model is trained on 64x64 multispectral images of Sentinel-2 satellite
data. Expected input is a 13-channel multispectral satellite image. See table
below for descriptions of the 13 bands. The ground sampling distanceshould be 
10 meters.

| Index | Band | Name |
|-------|------|------|
| 0 | B01 | Aerosols |
| 1 | B02 | Blue |
| 2 | B03 | Green |
| 3 | B04 | Red |
| 4 | B05 | Red Edge 1 |
| 5 | B06 | Red Edge 2 |
| 6 | B07 | Red Edge 3 |
| 7 | B08 | Near Infrared (NIR) |
| 8 | B08A | Narrow NIR |
| 9 | B09 | Water Vapour |
| 10 | B10 | Short-wave Infrared (SWIR) - Cirrus |
| 11 | B11 | SWIR 1 |
| 12 | B12 | SWIR 2 |

    """,
    version="alpha",
)

#@ Load models
model_rgb = get_model("models/model1.pkl")
model_ms = get_model("models/model2.pkl")
model_rgb.eval()
model_ms.eval()

## Functions for prediction API
def process_image(file: UploadFile, image_type: str) -> torch.Tensor:
    """    
    Pre-processes an image given by a user.
    Resizing (for RGB), normalization, and clipping extreme values (for MS).

    Args:
        file (UploadFile): User uploaded image or TIF file.
        image_type (str): File type. Options are "RGB" or "MS".

    Returns:
        torch.Tensor: Processed images as a tensor.
    """
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
    """
    Classifies a given image using a given model.

    Args:
        model (CNN): Model to use for prediction.
        image (torch.Tensor): Image to predict.

    Returns:
        ClassPredictions: Sorted predictions and confidence for each class.
    """
    with torch.no_grad():
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


## Functions for XAI API
def parse_target(target_class: int | str | None = None) -> int | None:
    """
    Parse target class for explainability from integer index or string.
    Default is None, for which explanation will be given for predicted class. 

    Args:
        target_class (int | str | None, optional): Target class to explain entered by user. 
                                                   Defaults to None.

    Raises:
        HTTPException: Invalid target class entered.

    Returns:
        int | None: Integer index for target class, or None.
    """
    if target_class is None:
        return None
    elif isinstance(target_class, int) and target_class in label_map:
        return target_class
    elif isinstance(target_class, str) and target_class in reverse_label_map:
        return reverse_label_map[target_class]

    raise HTTPException(
            status_code=400,
            detail="Invalid target_class. Valid options are integers 0-9 or "\
            "string representations from documentation."
        )


def class_to_explain(preprocessed_img: torch.Tensor, image_type: str) -> int:
    """
    Finds top prediction class for an input image using its corresponding type
    model.

    Args:
        preprocessed_img (torch.Tensor): Image to predict.
        image_type (str): File type. Options are "RGB" or "MS".

    Returns:
        int: Index of predicted class.
    """
    with torch.no_grad():
        if image_type == "RGB":
            predicted_class = int(model_rgb(preprocessed_img).argmax(1).item())
    
        elif image_type == "MS":
            predicted_class = int(model_ms(preprocessed_img).argmax(1).item())

    return predicted_class

def api_show_figures(figure: plt.Figure) -> StreamingResponse:
    """
    Transforms plt figure into StreamingResponse object for displaying images
    in FastAPI.

    Args:
        figure (plt.Figure): Figure to display.

    Returns:
        StreamingResponse: StreamingResponse object with figure to display.
    """
    # Save figure to buffer
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


def combine_xai_figures(ig_fig: plt.Figure, gradcam_fig: plt.Figure) -> plt.Figure:
    """
    Combines Integrated Gradients and GradCAM XAI figures into one figure in a
    side-by-side layout.

    Args:
        ig_fig (plt.Figure): Integrated Gradients figure.
        gradcam_fig (plt.Figure): GradCAM figure.

    Returns:
        plt.Figure: Combined figure.
    """
    ig_buf = io.BytesIO()
    ig_fig.savefig(ig_buf, format="png")
    plt.close(ig_fig)

    gradcam_buf = io.BytesIO()
    gradcam_fig.savefig(gradcam_buf, format="png")
    plt.close(gradcam_fig)

    ig_buf.seek(0)
    gradcam_buf.seek(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(plt.imread(ig_buf))
    ax1.axis("off")
    ax1.set_title("Integrated Gradients")
    ax2.imshow(plt.imread(gradcam_buf))
    ax2.axis("off")
    ax2.set_title("GradCAM")
    plt.tight_layout()

    return fig


def ig_explain(raw: torch.Tensor, preprocessed: torch.Tensor, baseline: torch.Tensor, predicted_class: int, target_class: int | None, n_steps: int, image_type: str) -> plt.Figure:
    """
    Performs Integrated Gradients on input image for a requested target class
    and returns visualization.

    Args:
        raw (torch.Tensor): [3, H, W] float tensor in [0, 1]
        preprocessed (torch.Tensor): Preprocessed input [C, H, W].
        baseline (torch.Tensor): Reference input [C, H, W], typically all-zeros.
        predicted_class (int): Class predicted by the model.
        target_class (int | None): Class being explained.
        n_steps (int): Number of interpolation steps (more = more accurate).
        image_type (str): File type. Options are "RGB" or "MS".

    Returns:
        plt.Figure: Visualization of Integrated Gradients explanation.
    """
    if image_type == "RGB":
        attrs = integrated_gradients(model_rgb, preprocessed, baseline, target_class, n_steps)
        figure = visualise_rgb(raw, attrs, predicted_class, target_class, output_path=None)
    elif image_type == "MS":
        attrs = integrated_gradients(model_ms, preprocessed, baseline, target_class, n_steps)
        figure = visualise_ms(raw, attrs, predicted_class, target_class, output_path=None)

    return figure

def gradcam_explain(img_tensor: torch.Tensor, img_array: np.ndarray, predicted_class: int, target_class: int | None, image_type: str) -> plt.Figure:
    """
    Performs GradCAM on input image for a requested target class and returns 
    visualization.    

    Args:
        img_tensor (torch.Tensor): Tensor of image to explain.
        img_array (np.ndarray): Array of image to explain.
        predicted_class (int): Class predicted by the model.
        target_class (int | None): Class being explained.
        image_type (str): File type. Options are "RGB" or "MS".

    Returns:
        plt.Figure: Visualization of GradCAM explanation.
    """
    if image_type == "RGB":
        gradcam_visualization = gradcam(model_rgb, img_tensor, img_array, target_class)
    elif image_type == "MS":
        gradcam_visualization = gradcam(model_ms, img_tensor, img_array, target_class)
    
    figure, ax = plt.subplots()
    ax.imshow(gradcam_visualization)
    ax.set_title(f"GradCAM | Predicted: {label_map[predicted_class]} | Explaining: {label_map[target_class]}")
    ax.axis("off")
    return figure

## API endpoints
@app.get("/", description="Root endpoint that redirects to API documentation.")
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
async def predict_rgb(image: UploadFile) -> ClassPredictions:
    # For final API version, create something
    # that can handle more than 10x10 meter images. TODO
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
async def predict_ms(image: UploadFile) -> ClassPredictions:
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
    summary="Provide explainability for RGB model on input image for desired " 
    "class.",
    description="RGB explainability endpoint to give insight into model "
    "responses for RGB inputs. Requests should be of the format multipart/form-data, "
    "and include an image sent using the applicable 'image' field. The given "
    "image should be a three-channel RGB satellite image using a 10m ground "
    "sampling distance. That is, the distance between the center of two "
    "consecutive pixels is 10m when measured on the ground. Optionally, a "
    "target class may be included in the request. This should either be an "
    "integer between 0 and 9 included, or a class name as written in the "
    "table above. When no target class is provided, the explanation of the "
    "model will be given for the predicted class. Additionally, an n_steps "
    "input can be included in the request. This refers to the number of "
    "interpolation steps for integrated gradients, and higher numbers "
    "generally create more accurate results. The default value is 50. Returns "
    "a double panel: on the left, the Integrated Gradients explanation is "
    "shown alongside the original input, where brighter colors indicate more " 
    "influential pixels; on the right, the GradCAM explanation is shown "
    "alongside the original input, and red colors indicate more influential "
    "pixels.",
    response_description="Returns Integrated Gradients attribution heatmap " 
    "and GradCAM heatmap for specified target class.",
    response_class = StreamingResponse,
)
async def explain_rgb(image: UploadFile, target_class: int | None = None, n_steps: int = 50) -> StreamingResponse:
    try:
        # GradCam inputs
        tensor_image = process_image(image, "RGB") # dim (1, 3, H, W)
        array_image = tensor_image[0].permute(1, 2, 0).numpy() 

        # Integrated gradients inputs
        preprocessed = tensor_image[0]
        raw = preprocessed
        baseline = torch.zeros_like(preprocessed)

    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")

    target_class = parse_target(target_class)
    predicted_class = class_to_explain(tensor_image, "RGB")
    if target_class is None:
        target_class = predicted_class
    
    ig_fig = ig_explain(raw, preprocessed, baseline, predicted_class, target_class, n_steps, image_type="RGB")
    gradcam_fig = gradcam_explain(tensor_image, array_image, predicted_class, target_class, image_type="RGB")

    return api_show_figures(combine_xai_figures(ig_fig, gradcam_fig))




# Explainability for MS model - GradCAM & Integrated Gradients
@app.post(
    "/explain_ms",
    summary="Provide explainability for MS model on input image for desired " 
    "class.",
    description="MS explainability endpoint to give insight into model "
    "responses for MS inputs. Requests should be of the format multipart/form-data, "
    "and include an image sent using the applicable 'image' field. The given "
    "image should be a thirteen-channel multispectral satellite image using a "
    "10m ground sampling distance. That is, the distance between the center of two "
    "consecutive pixels is 10m when measured on the ground. Optionally, a "
    "target class may be included in the request. This should either be an "
    "integer between 0 and 9 included, or a class name as written in the "
    "table above. When no target class is provided, the explanation of the "
    "model will be given for the predicted class. Additionally, an n_steps "
    "input can be included in the request. This refers to the number of "
    "interpolation steps for integrated gradients, and higher numbers "
    "generally create more accurate results. The default value is 50. Returns "
    "a double panel: on the left, the Integrated Gradients explanation is "
    "shown (per-band and aggregated) alongside the original input, where " 
    "brighter colors indicate more influential pixels; on the right, the "
    "GradCAM explanation is shown alongside the original input, and red colors " 
    "indicate more influential pixels.",    
    response_description="Returns Integrated Gradients attribution heatmap " 
    "and GradCAM heatmap for specified target class. Integrated Gradients "
    "visualization contains 15 heatmaps: \n"
    " - Cell 0: RGB composite (R=B4, G=B3, B=B2) for visual context."
    " - Cells 1-13: Per-band attribution maps using a diverging red/blue colormap:"
    "       Red = pushed model toward the class"
    "       Blue = pushed model away from the class"
    " - Cell 14: Aggregate attribution (sum of absolute values across all bands)",
    response_class = StreamingResponse,
)
async def explain_ms(image: UploadFile, target_class: int | None = None, n_steps: int = 50) -> StreamingResponse:
    try:
        tif_bytes = image.file.read()

        raw = tifffile.imread(io.BytesIO(tif_bytes))
        raw = torch.from_numpy(raw).permute(2, 0, 1)
        tensor_image = normalize_MS_img(raw).unsqueeze(0) # dim: (1, 13, H, W)
        preprocessed = tensor_image[0]
        baseline = torch.zeros_like(preprocessed)
        array_image = _scaled_rgb_colour(raw)

    except tifffile.tifffile.TiffFileError:
        raise HTTPException(
            status_code=415,
            detail="Invalid image extension specified. "
            "Multispectral image prediction accepts "
            "only TIF files.",
        )
    target_class = parse_target(target_class)
    predicted_class = class_to_explain(preprocessed.unsqueeze(0), "MS")
    if target_class is None:
        target_class = predicted_class

    ig_fig = ig_explain(raw, preprocessed, baseline, predicted_class, target_class, n_steps, image_type="MS")
    gradcam_fig = gradcam_explain(tensor_image, array_image, predicted_class, target_class, image_type="MS")

    return api_show_figures(combine_xai_figures(ig_fig, gradcam_fig))
