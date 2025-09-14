# Copyright (C) 2024 Prashant-b97
# AGPL-3.0-licensed

import os
import datetime
import sys
import argparse
from ultralytics import YOLO
from typing import List, Any

def print_detections(results: List[Any]):
    """Prints a summary of the detection results."""
    # The results object contains lots of information. We'll just print a summary.
    # Assumes results are for a single image.
    result = results[0]
    print(f"\nOutput image saved in: {result.save_dir}")
    print(f"Total objects detected: {len(result.boxes)}")
    
    if not len(result.boxes):
        print("\n- No objects detected.")
        return

    print("\n--- Detected Objects ---")
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        confidence = float(box.conf[0])
        # Get bounding box coordinates in (top, left, bottom, right) format
        coords = box.xyxy[0].tolist()
        coords = [round(x) for x in coords]
        print(f"- Class: {class_name} ({confidence:.2%})")
        print(f"  - Bounding Box: [x1: {coords[0]}, y1: {coords[1]}, x2: {coords[2]}, y2: {coords[3]}]")
    print("------------------------")


def detect_objects(args: argparse.Namespace):
    """Handles object detection using YOLOv8."""
    model_path = args.model
    input_image_path = args.input

    if not os.path.exists(input_image_path):
        sys.stderr.write(f"Error: Input image not found at {input_image_path}\n")
        return

    try:
        print("Loading YOLOv8 model...")
        model = YOLO(model_path)

        print("Starting object detection...")
        # Generate a unique name for the output directory using a timestamp
        run_name = f"detect_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Run inference. The library handles drawing boxes and saving the image.
        results = model.predict(
            source=input_image_path,
            save=True,
            conf=args.probability / 100.0,  # Convert percentage to 0-1 scale
            project=args.output_dir,
            name=run_name,
            exist_ok=True  # Allow writing to an existing directory
        )
        print("Detection complete.")
        print_detections(results)
    except Exception as e:
        sys.stderr.write(f"An error occurred during detection: {e}\n")


def train_model(args: argparse.Namespace):
    """Handles model training using YOLOv8."""

    print("Starting model training...")
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
        sys.stderr.write(f"An error occurred during training: {e}\n")
    print("Training complete. The best model is saved in the 'runs/train/...' directory.")

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

    args = parser.parse_args()

    # If no command is provided, print help and exit.
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()
