from fastapi import FastAPI
from starlette.responses import RedirectResponse

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


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url="/docs")
