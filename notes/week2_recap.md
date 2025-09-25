# Week 2 Recap — Explainability & UX

## Highlights
- **Explainability with Grad-CAM**: Implemented Grad-CAM to generate heatmaps showing model attention. This involved adding a `detect_with_heatmap` method to the core detector, using PyTorch hooks to capture gradients, and performing a manual forward pass to enable gradient computation during inference.
- **Gradio Web UI**: Built a user-friendly web interface (`app.py`) for easy interaction. It supports:
  - Image detection for objects and poses.
  - A checkbox to toggle Grad-CAM visualizations.
  - A "Video Detection" tab for batch processing.
  - A new checkbox to run combined object detection and pose estimation on videos.
- **FastAPI REST API**: Created a high-performance API (`api.py`) with a `/detect` endpoint. It accepts an image and an optional `conf_threshold` and returns structured JSON, making the model accessible as a microservice.
- **Unit Testing**: Added tests for the new API endpoint (`tests/test_api.py`) and the core Grad-CAM logic (`tests/test_core.py`) to ensure stability and prevent regressions.

## Testing
- All unit tests pass via `python -m unittest discover`.
- Manual validation confirmed the Grad-CAM overlay and combined video processing work correctly in the Gradio UI.