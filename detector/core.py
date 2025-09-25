from __future__ import annotations

from typing import List, NamedTuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f


def _letterbox_image(
    image: np.ndarray,
    target_size: int,
    stride: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Resize and pad image to keep aspect ratio while meeting stride multiples."""
    height, width = image.shape[:2]
    if isinstance(target_size, int):
        target_shape = (target_size, target_size)
    else:
        target_shape = target_size

    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_unpadded = (int(round(width * scale)), int(round(height * scale)))

    dw = target_shape[1] - new_unpadded[0]
    dh = target_shape[0] - new_unpadded[1]

    dw /= 2
    dh /= 2

    if (width, height) != new_unpadded:
        resized = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)
    else:
        resized = image.copy()

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, (top, bottom, left, right), scale


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
        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # pragma: no cover - just-in-case defensive block
            raise ValueError(f"Error loading model from {model_path}: {exc}") from exc

    def detect_from_image(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> List[Detection]:
        """Performs object detection on a single image."""
        results = self.model.predict(image, conf=conf_threshold, verbose=False)

        detections: List[Detection] = []
        result = results[0]
        names = result.names
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            bounding_box = BoundingBox(
                x1=round(coords[0]),
                y1=round(coords[1]),
                x2=round(coords[2]),
                y2=round(coords[3]),
            )
            detections.append(
                Detection(
                    class_name=names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=bounding_box,
                )
            )

        return detections

    def detect_with_heatmap(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> np.ndarray:
        """Perform detection and overlay a Grad-CAM heatmap highlighting model focus."""
        feature_maps: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_: torch.nn.Module, __: tuple, output: torch.Tensor) -> None:
            feature_maps.append(output)

        def backward_hook(_: torch.nn.Module, __: tuple, grad_out: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_out[0])

        # Locate the last C2f block of the backbone for Grad-CAM
        target_layer: Optional[torch.nn.Module] = None
        for module in reversed(self.model.model.model[:10]):
            if isinstance(module, C2f):
                target_layer = module
                break

        if target_layer is None:
            detections = self.detect_from_image(image, conf_threshold)
            return draw_detections(image, detections)

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            detections = self.detect_from_image(image, conf_threshold)
            if not detections:
                return draw_detections(image, detections)

            # Determine which class to backpropagate – use highest confidence detection
            top_detection = max(detections, key=lambda det: det.confidence)

            names = getattr(self.model.model, "names", self.model.names)
            if isinstance(names, dict):
                class_index = next(key for key, value in names.items() if value == top_detection.class_name)
            else:
                class_index = names.index(top_detection.class_name)

            model_args = getattr(self.model.model, "args", {}) or {}
            imgsz = int(model_args.get("imgsz", 640))
            stride = int(getattr(self.model.model, "stride", torch.tensor([32])).max())

            padded_image, pads, _ = _letterbox_image(image.astype(np.uint8), target_size=imgsz, stride=stride)
            tensor_input = (
                torch.from_numpy(padded_image.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .div(255.0)
                .to(self.model.device)
            )
            tensor_input.requires_grad_(True)

            with torch.enable_grad():
                model_outputs = self.model.model(tensor_input)

            detection_output = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs
            class_scores = detection_output[..., 4 + class_index]
            target = class_scores.max()
            target.backward(retain_graph=True)
            self.model.model.zero_grad(set_to_none=True)

            if not gradients or not feature_maps:
                return draw_detections(image, detections)

            pooled_gradients = torch.mean(gradients[0], dim=[2, 3])
            weighted_features = feature_maps[0] * pooled_gradients.view(1, -1, 1, 1)
            heatmap = torch.mean(weighted_features, dim=1).squeeze()
            heatmap = torch.relu(heatmap)
            max_val = torch.max(heatmap)
            if max_val > 0:
                heatmap /= max_val

            heatmap = heatmap.detach().cpu().numpy()
            heatmap_resized = cv2.resize(heatmap, (padded_image.shape[1], padded_image.shape[0]))

            top, bottom, left, right = pads
            bottom_idx = heatmap_resized.shape[0] - bottom if bottom > 0 else heatmap_resized.shape[0]
            right_idx = heatmap_resized.shape[1] - right if right > 0 else heatmap_resized.shape[1]
            cropped = heatmap_resized[top:bottom_idx, left:right_idx]
            heatmap_original = cv2.resize(cropped, (image.shape[1], image.shape[0]))

            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_original), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            image_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
            blended = cv2.addWeighted(image_uint8, 0.6, heatmap_rgb, 0.4, 0)
            return draw_detections(blended, detections)
        finally:
            forward_handle.remove()
            backward_handle.remove()


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw detection bounding boxes and labels on an image."""
    output_image = image.copy()
    for det in detections:
        cv2.rectangle(
            output_image,
            (det.box.x1, det.box.y1),
            (det.box.x2, det.box.y2),
            (36, 255, 12),
            2,
        )
        label = f"{det.class_name}: {det.confidence:.2%}"
        cv2.putText(
            output_image,
            label,
            (det.box.x1, det.box.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (36, 255, 12),
            2,
        )
    return output_image
