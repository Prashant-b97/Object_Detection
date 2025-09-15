# Copyright (C) 2024 Prashant-b97
# AGPL-3.0-licensed

import os
import datetime
import logging
import sys
import argparse
import cv2
from ultralytics import YOLO
from typing import List

# Import from our new core library
from detector.core import ObjectDetector, Detection, draw_detections

def print_detections(detections: List[Detection], output_dir: str):
    """Prints a summary of the detection results."""
    logging.info(f"Output image saved in: {output_dir}")
    logging.info(f"Total objects detected: {len(detections)}")
    
    if not detections:
        logging.info("- No objects detected.")
        return

    detection_summary = ["--- Detected Objects ---"]
    for det in detections:
        detection_summary.append(f"- Class: {det.class_name} ({det.confidence:.2%})")
        detection_summary.append(f"  - Bounding Box: [x1: {det.box.x1}, y1: {det.box.y1}, x2: {det.box.x2}, y2: {det.box.y2}]")
    detection_summary.append("------------------------")
    logging.info("\n".join(detection_summary))

def detect_objects(args: argparse.Namespace):
    """Handles object detection using YOLOv8."""
    model_path = args.model
    input_image_path = args.input

    if not os.path.exists(input_image_path):
        logging.error(f"Input image not found at {input_image_path}")
        return

    try:
        # 1. Initialize our core detector
        detector = ObjectDetector(model_path)

        # 2. Read the image
        image = cv2.imread(input_image_path)
        if image is None:
            logging.error(f"Could not read image file at {input_image_path}")
            return

        # 3. Perform detection using the core library
        logging.info("Starting object detection...")
        confidence = args.probability / 100.0
        detections = detector.detect_from_image(image, conf_threshold=confidence)
        logging.info("Detection complete.")

        # 4. Draw detections on the image
        output_image = draw_detections(image, detections)

        # 5. Save the output image
        run_name = f"detect_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(input_image_path))
        cv2.imwrite(output_path, output_image)

        # 6. Print results to the console
        print_detections(detections, output_dir)
    except Exception as e:
        logging.error(f"An error occurred during detection: {e}", exc_info=True)


def train_model(args: argparse.Namespace):
    """Handles model training using YOLOv8."""

    logging.info("Starting model training...")
    # Load a pretrained model (e.g., yolov8n.pt) to start training from
    model = YOLO(args.pretrained_model)

    # Train the model
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=640,  # Image size, a common default for YOLOv8
            project="runs/train",
            name=f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
    logging.info("Training complete. The best model is saved in the 'runs/train/...' directory.")

def evaluate_model(args: argparse.Namespace):
    """Handles model evaluation using YOLOv8 and prints performance metrics."""
    model_path = args.model

    if model_path == 'latest':
        logging.info("Finding the latest trained model...")
        train_dir = 'runs/train'
        if not os.path.isdir(train_dir):
            logging.error(f"Training directory '{train_dir}' not found. Please train a model first.")
            return

        all_runs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('train')]
        if not all_runs:
            logging.error(f"No training runs found in '{train_dir}'.")
            return

        latest_run = sorted(all_runs)[-1]
        model_path = os.path.join(train_dir, latest_run, 'weights', 'best.pt')

        if not os.path.exists(model_path):
            logging.error(f"'best.pt' not found in the latest training run: {os.path.join(train_dir, latest_run)}")
            return
        
        logging.info(f"Found latest model: {model_path}")

    logging.info(f"Loading model for evaluation: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}", exc_info=True)
        return

    logging.info(f"Starting evaluation on dataset specified in: {args.data}")
    try:
        # The val() method runs evaluation and prints a comprehensive table of metrics.
        metrics = model.val(data=args.data)
        logging.info("Evaluation complete. See the metrics table above for performance details (mAP, Precision, Recall).")
        # The metrics object itself contains detailed data if you want to process it further.
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(
        description="A tool for training and running YOLOv8 object detection models."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Parser for the 'detect' command ---
    parser_detect = subparsers.add_parser("detect", help="Detect objects in an image with a trained YOLO model.")
    parser_detect.add_argument(
        "-i", "--input", default="sample_input/image.jpg", help="Path to the input image file."
    )
    parser_detect.add_argument(
        "-m", "--model", required=True, help="Path to the trained YOLO model file (.pt)."
    )
    parser_detect.add_argument(
        "-o", "--output-dir", default="runs/detect", help="Directory to save the output."
    )
    parser_detect.add_argument(
        "-p", "--probability", type=float, default=25, help="Minimum detection confidence threshold (0-100)."
    )
    parser_detect.set_defaults(func=detect_objects)

    # --- Parser for the 'train' command ---
    parser_train = subparsers.add_parser("train", help="Train a new YOLOv8 object detection model.")
    parser_train.add_argument(
        "--data", required=True, help="Path to the dataset YAML file (e.g., coco8.yaml)."
    )
    parser_train.add_argument(
        "--pretrained-model", default="yolov8n.pt",
        help="Path to the pretrained model to use for transfer learning (e.g., yolov8n.pt, yolov8s.pt)."
    )
    parser_train.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser_train.add_argument(
        "--batch-size", type=int, default=8, help="Training batch size. Adjust based on your GPU/CPU memory."
    )
    parser_train.set_defaults(func=train_model)

    # --- Parser for the 'evaluate' command ---
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model's performance on a dataset.")
    parser_eval.add_argument(
        "--data", required=True, help="Path to the dataset YAML file (e.g., coco8.yaml)."
    )
    parser_eval.add_argument(
        "-m", "--model", required=True, help="Path to the trained YOLO model file (.pt) to evaluate."
    )
    parser_eval.set_defaults(func=evaluate_model)

    args = parser.parse_args()

    # If no command is provided, print help and exit.
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # --- Setup Logging ---
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )

    args.func(args)

if __name__ == "__main__":
    main()
