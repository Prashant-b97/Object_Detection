# Copyright (C) 2024 Prashant-b97
# AGPL-3.0-licensed

import cv2
import os
import sys
import logging
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import from our new core library
# NOTE: We will use the library's built-in plot() for flexibility with different model types.
# from detector.core import ObjectDetector, draw_detections

def process_video(
    model: YOLO,
    source: any,
    conf_threshold: float,
    output_path: str = None,
    enable_tracking: bool = False,
    max_frames: int = 0,
    frame_skip: int = 0,
):
    """
    Processes a video source (file or webcam) for object detection or pose estimation.

    Args:
        model (YOLO): The YOLO model instance.
        source (int or str): The video source (0 for webcam, or path to video file).
        conf_threshold (float): The confidence threshold for detection (0-1).
        output_path (str, optional): Path to save the output video. Defaults to None.
        enable_tracking (bool, optional): Whether to enable DeepSORT object tracking. Defaults to False.
        max_frames (int, optional): Maximum frames to process. 0 means no limit.
        frame_skip (int, optional): Number of frames to skip between processed frames.
    """
    # First, check if the video source is a file and if it exists.
    # This provides a more specific error than OpenCV's generic one.
    if isinstance(source, str):
        if not os.path.exists(source):
            logging.error(f"Input video file not found at: '{source}'")
            return
        # Add a file size check to catch invalid downloads.
        # A valid video file will be larger than a few kilobytes.
        if os.path.getsize(source) < 1024: # 1 KB
            logging.error(f"Input video file at '{source}' is too small to be a valid video. Please check the download.")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(f"Could not open video source '{source}'. It may be an invalid file or a device that is not available.")
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

    # Initialize the DeepSORT tracker if enabled
    tracker = None
    if enable_tracking:
        tracker = DeepSort(max_age=30)
        logging.info("DeepSORT tracking enabled.")
        # For drawing unique colors per track
        track_colors = {}

    frame_idx = -1
    processed_frame_count = 0

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break

        # Update progress for batch mode for every frame read
        if pbar:
            pbar.update(1)

        # Frame skipping: process only every (frame_skip + 1)th frame
        if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
            continue

        # Stop when reaching max_frames processed frames
        if max_frames > 0 and processed_frame_count >= max_frames:
            logging.info(f"Reached max-frames limit of {max_frames}. Stopping processing.")
            break

        processed_frame_count += 1

        # Run YOLOv8 inference on the frame. The library handles both detection and pose.
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        if enable_tracking and tracker is not None:
            # Prepare detections for the tracker
            detections_for_tracker = []
            for box in results[0].boxes.data:
                # box format is [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2, conf, cls = box.tolist()
                # DeepSORT expects [x, y, w, h]
                w, h = x2 - x1, y2 - y1
                # We also need the class name for potential filtering, though not used here
                class_name = model.names[int(cls)]
                
                # For now, we track all detected objects.
                # You could filter by class_name here if needed.
                detections_for_tracker.append(([x1, y1, w, h], conf, class_name))

            # Update the tracker with the new detections
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

            # Draw the tracks on the frame
            annotated_frame = frame.copy()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb() # Bbox in [x1, y1, x2, y2] format
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

                # Assign a unique color to each track ID
                if track_id not in track_colors:
                    track_colors[track_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_colors[track_id], 2)
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_colors[track_id], 2)
        else:
            # Visualize the results on the frame using the library's built-in plotter.
            annotated_frame = results[0].plot()

        # Write the frame to the output video if a writer is initialized
        if video_writer:
            video_writer.write(annotated_frame)

        if not is_batch_mode:
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
    parser.add_argument(
        '--enable-tracking', action='store_true', help="Enable object tracking with DeepSORT."
    )
    parser.add_argument(
        '--max-frames', type=int, default=0,
        help="Maximum number of frames to process. 0 for unlimited."
    )
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help="Number of frames to skip between processed frames (e.g., 1 means process every 2nd frame)."
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

    process_video(
        model,
        video_source,
        confidence,
        output_path=output_full_path,
        enable_tracking=args.enable_tracking,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )

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
