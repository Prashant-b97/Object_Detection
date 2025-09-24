"""Core detector package exposing YOLO-based detection helpers."""

from .core import ObjectDetector, Detection, BoundingBox, draw_detections

__all__ = [
    "ObjectDetector",
    "Detection",
    "BoundingBox",
    "draw_detections",
]
