from __future__ import annotations

import gradio as gr
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detector.core import ObjectDetector, draw_detections
from scripts.videodetection import VideoDetector

EVAL_DASHBOARD_URL = "http://localhost:8501"

# --- Configuration ---
DEFAULT_MODEL = "yolov8n.pt"
POSE_MODEL = "yolov8n-pose.pt"
MIN_DISPLAY_CONF = 0.10  # Keep overlays readable even if slider is set very low
MODELS = {}  # Dictionary to cache loaded models


def get_model(model_name: str, model_path: str):
    """Loads a model if not already in the cache, then returns it."""
    if model_name not in MODELS:
        print(f"Loading {model_name} model from {model_path}...")
        MODELS[model_name] = ObjectDetector(model_path=model_path)
        print(f"{model_name} model loaded.")
    return MODELS[model_name]


def get_video_detector():
    """Loads the VideoDetector if not already in the cache."""
    if "video_detector" not in MODELS:
        print("Loading VideoDetector...")
        MODELS["video_detector"] = VideoDetector(model_path=DEFAULT_MODEL, pose_model_path=POSE_MODEL)
        print("VideoDetector loaded.")
    return MODELS["video_detector"]

# --- Processing Functions ---
def process_image(
    image: np.ndarray, model_choice: str, show_heatmap: bool, conf_threshold: float
) -> np.ndarray:
    """Process a single image for object detection or pose estimation."""
    if image is None:
        return None

    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 4: # RGBA
        image = image[:, :, :3]

    # Load the appropriate model on demand
    model_path = POSE_MODEL if model_choice == "Pose" else DEFAULT_MODEL
    active_detector = get_model(model_choice, model_path)

    conf = conf_threshold / 100.0  # Convert from percentage
    if conf < MIN_DISPLAY_CONF:
        conf = MIN_DISPLAY_CONF
    if show_heatmap and model_choice == "Objects":
        # Grad-CAM is typically more insightful for object detection.
        return active_detector.detect_with_heatmap(image, conf_threshold=conf)
    else:
        detections = active_detector.detect_from_image(image, conf_threshold=conf)
        return draw_detections(image, detections)


def process_video_ui(video_path: str, enable_pose: bool):
    """Process a video and return path to the annotated copy."""
    if not video_path:
        return None

    # Load the video detector on demand
    video_detector = get_video_detector()

    output_full_path = video_detector.process_video_batch(
        video_path,
        conf_threshold=0.25,
        enable_pose=enable_pose,
    )

    return output_full_path


# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Soft(), title="YOLOv8 Object & Pose Detector"
) as demo:
    # State object to hold loaded models across requests
    gr.Markdown(
        "# 🚀 YOLOv8 Object & Pose Detector\n"
        "Upload an image or video to see the model in action."
    )

    with gr.Tab("🖼️ Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload Image")
                model_choice = gr.Radio(
                    ["Objects", "Pose"],
                    value="Objects",
                    label="Detection Type",
                    info="Choose 'Pose' for human keypoint detection.",
                )
                conf_slider = gr.Slider(
                    minimum=0, maximum=100, value=25, step=1,
                    label="Confidence Threshold (%)",
                    info="Minimum confidence for a detection to be shown. Values below 10% are clamped to keep results readable."
                )
                show_heatmap = gr.Checkbox(
                    label="Show Grad-CAM Heatmap",
                    value=False,
                    info="Visualize where the model is 'looking'. (Object detection only)",
                )
                image_button = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Result")

    with gr.Tab("🎬 Video Detection (Batch Mode)"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                pose_checkbox = gr.Checkbox(
                    label="Enable Simultaneous Pose Estimation",
                    value=False,
                    info="Detects human poses along with objects. May be slower."
                )
                video_button = gr.Button("Process Video", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Result")

    with gr.Tab("📊 Evaluation Dashboard"):
        gr.Markdown(
            """
            ### Review Training Metrics

            Launch the Streamlit dashboard to inspect mAP/precision/recall trends, PR curves, and FPS benchmarks.

            1. In a terminal, run:
               ```bash
               streamlit run services/streamlit/dashboard.py
               ```
            2. Once running, open the dashboard here:
               - [Open Evaluation Dashboard]({url})

            The dashboard reads experiment data from `reports/metrics.json`, so new evaluations will appear automatically once logged.
            """.format(url=EVAL_DASHBOARD_URL)
        )

    # --- Event Handlers ---
    image_button.click(
        fn=process_image,
        inputs=[image_input, model_choice, show_heatmap, conf_slider],
        outputs=image_output,
    )
    video_button.click(
        fn=process_video_ui, inputs=[video_input, pose_checkbox], outputs=video_output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
