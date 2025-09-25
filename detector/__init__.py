# This file makes the 'detector' directory a Python package.

from .core import ObjectDetector, Detection, BoundingBox, draw_detections

__all__ = ["ObjectDetector", "Detection", "BoundingBox", "draw_detections"]
