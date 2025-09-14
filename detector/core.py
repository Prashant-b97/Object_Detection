from ultralytics import YOLO
from typing import List, NamedTuple
import numpy as np

class BoundingBox(NamedTuple):
    """Represents a bounding box with coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(NamedTuple):
    """Represents a single detected object."""
    class_name: str
    confidence: float
    box: BoundingBox

class ObjectDetector:
    """A class to encapsulate the YOLO model and detection logic."""

    def __init__(self, model_path: str):
        """
        Initializes the detector by loading the YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model file (.pt).
        """
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            # Raise a more specific error for the application layer to handle.
            raise ValueError(f"Error loading model from {model_path}: {e}") from e

    def detect_from_image(self, image: np.ndarray, conf_threshold: float) -> List[Detection]:
        """
        Performs object detection on a single image.

        Args:
            image (np.ndarray): The image to process (as a NumPy array).
            conf_threshold (float): The confidence threshold for detection.

        Returns:
            List[Detection]: A list of detected objects.
        """
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        result = results[0]
        names = result.names
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            bounding_box = BoundingBox(x1=round(coords[0]), y1=round(coords[1]), x2=round(coords[2]), y2=round(coords[3]))
            detection = Detection(class_name=names[int(box.cls[0])], confidence=float(box.conf[0]), box=bounding_box)
            detections.append(detection)
            
        return detections
