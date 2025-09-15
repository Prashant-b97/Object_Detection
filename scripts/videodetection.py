# Copyright (C) 2024 Prashant-b97
# AGPL-3.0-licensed

import cv2
import os
import sys
import logging
import datetime
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# Import from our new core library
# NOTE: We will use the library's built-in plot() for flexibility with different model types.
# from detector.core import ObjectDetector, draw_detections

def process_video(model: YOLO, source: any, conf_threshold: float, output_path: str = None):
    """
    Processes a video source (file or webcam) for object detection or pose estimation.

    Args:
        model (YOLO): The YOLO model instance.
        source (int or str): The video source (0 for webcam, or path to video file).
        conf_threshold (float): The confidence threshold for detection (0-1).
        output_path (str, optional): Path to save the output video. Defaults to None.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"Could not open video source '{source}'")
        return
    
    # Determine if we are in interactive mode or batch processing mode
    is_webcam = isinstance(source, int) and source == 0
    # Batch mode is when processing a file and saving the output
    is_batch_mode = not is_webcam and output_path is not None

    if is_batch_mode:
        logging.info("Running in batch mode (processing file without display)...")
    else:
        logging.info("Starting video processing... Press 'q' to quit.")

    # Get video properties for the writer
    video_writer = None
    if output_path:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Attempting to save output video to: {output_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Use a default FPS if it's a webcam stream, which often returns 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20  # A reasonable default for webcams

        # Use 'avc1' (H.264) as it's more broadly compatible, especially on macOS, than 'mp4v'.
        # Other options include 'mp4v', or 'XVID' for .avi files.
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not video_writer.isOpened():
            logging.error(f"Could not open video writer for path: {output_path}")
            # We'll continue with displaying the video, but saving will be disabled.
            video_writer = None

    # Setup progress bar for batch mode
    pbar = None
    if is_batch_mode:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing video frames")

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        # Run YOLOv8 inference on the frame. The library handles both detection and pose.
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        # Visualize the results on the frame using the library's built-in plotter.
        annotated_frame = results[0].plot()

        # Write the frame to the output video if a writer is initialized
        if video_writer:
            video_writer.write(annotated_frame)

        if is_batch_mode:
            pbar.update(1)
        else:
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    if pbar:
        pbar.close()
        logging.info("Batch processing complete.")
        
    # Release the video capture object and close the display window
    cap.release()
    if video_writer:
        video_writer.release()
        logging.info(f"Output video saved to: {output_path}")
    cv2.destroyAllWindows()
    logging.info("Video processing finished.")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on a video or webcam.")
    parser.add_argument(
        "-m", "--model", required=True, help="Path to the trained YOLO model file (.pt)."
    )
    parser.add_argument(
        "-i", "--input", default="0",
        help="Path to the input video file or '0' for webcam. Defaults to '0' (webcam)."
    )
    parser.add_argument(
        "-p", "--probability", type=float, default=25,
        help="Minimum detection confidence threshold (0-100). Default is 25."
    )
    parser.add_argument(
        "-o", "--output", help="Directory to save the output video. If provided, a timestamped video will be saved."
    )
    args = parser.parse_args()

    # Convert source to integer if it's the webcam
    video_source = 0 if args.input == '0' else args.input
    if video_source != 0:
        logging.info(f"Input source: {video_source}")
    else:
        logging.info("Input source: Webcam")

    # Generate a unique output path if an output directory is provided
    output_full_path = None
    if args.output:
        output_dir = args.output
        
        # Create a unique filename based on source and timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if video_source == 0:
            base_name = "webcam_output"
        else:
            # Use the name of the input file without its extension
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            
        output_filename = f"{base_name}_{timestamp}.mp4"
        output_full_path = os.path.join(output_dir, output_filename)

    # Load the YOLOv8 model
    logging.info(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return

    # Convert confidence from 0-100 to 0-1
    confidence = args.probability / 100.0

    process_video(model, video_source, confidence, output_path=output_full_path)

def setup_logging():
    """Configures the logging for the application."""
    # Check if handlers are already configured to prevent re-configuration.
    if logging.getLogger().hasHandlers():
        return

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

if __name__ == "__main__":
    setup_logging()
    main()
