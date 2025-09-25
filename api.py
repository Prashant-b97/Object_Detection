import io
from typing import List

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from detector.core import ObjectDetector

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="An API for performing object detection using a YOLOv8 model.",
    version="1.0.0",
)

# It's good practice to load the model once at startup.
# For simplicity here, we use a default model.
# In a production scenario, you might load this from a config file.
detector = ObjectDetector(model_path="yolov8n.pt")


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    box: BoundingBox
    class_name: str
    confidence: float


@app.post("/detect", response_model=List[DetectionResult])
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.25, ge=0.0, le=1.0)
):
    """
    Performs object detection on an uploaded image.

    - **file**: The image file to process.
    - **conf_threshold**: The confidence threshold for detections (0.0 to 1.0).

    Returns a list of detected objects with their bounding boxes,
    class names, and confidence scores.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    detections = detector.detect_from_image(image_np, conf_threshold=conf_threshold)

    results = [
        DetectionResult(
            box=BoundingBox(x1=d.box.x1, y1=d.box.y1, x2=d.box.x2, y2=d.box.y2),
            class_name=d.class_name,
            confidence=d.confidence,
        )
        for d in detections
    ]
    return results