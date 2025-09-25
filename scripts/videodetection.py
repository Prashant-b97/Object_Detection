# Copyright (C) 2024 Prashant-b97
# AGPL-3.0-licensed

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import setup_logging


POSE_MODEL_DEFAULT = "yolov8n-pose.pt"


class VideoDetector:
    """Run YOLO on videos with optional pose estimation and tracking."""

    def __init__(self, model_path: str, pose_model_path: Optional[str] = None):
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        self._model: Optional[YOLO] = None
        self._pose_model: Optional[YOLO] = None

    def _load_detection_model(self) -> YOLO:
        if self._model is None:
            logging.info("Loading detection model: %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def _load_pose_model(self) -> Optional[YOLO]:
        if not self.pose_model_path:
            return None
        if self._pose_model is None:
            logging.info("Loading pose model: %s", self.pose_model_path)
            self._pose_model = YOLO(self.pose_model_path)
        return self._pose_model

    @staticmethod
    def _generate_output_path(source_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if source_path in {"0", 0, None}:
            base_name = "webcam_output"
        else:
            base_name = os.path.splitext(os.path.basename(str(source_path)))[0]
        return os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")

    def process_video_batch(
        self,
        video_path: str,
        output_dir: str = "runs/detect_video",
        conf_threshold: float = 0.25,
        enable_pose: bool = False,
    ) -> str:
        output_full_path = self._generate_output_path(video_path, output_dir)
        detect_model = self._load_detection_model()
        pose_model = self._load_pose_model() if enable_pose else None
        process_video(
            model=detect_model,
            source=video_path,
            conf_threshold=conf_threshold,
            output_path=output_full_path,
            enable_tracking=False,
            max_frames=0,
            frame_skip=0,
            pose_model=pose_model,
            pose_conf_threshold=conf_threshold,
        )
        return output_full_path

    def process_video_interactive(self, **kwargs) -> None:
        detect_model = self._load_detection_model()
        enable_pose = kwargs.pop("enable_pose", False)
        kwargs.setdefault("pose_model", self._load_pose_model() if enable_pose else None)
        process_video(model=detect_model, **kwargs)


def process_video(
    model: YOLO,
    source: int | str,
    conf_threshold: float,
    output_path: Optional[str] = None,
    enable_tracking: bool = False,
    max_frames: int = 0,
    frame_skip: int = 0,
    pose_model: Optional[YOLO] = None,
    pose_conf_threshold: Optional[float] = None,
) -> None:
    """Process a video source for detection/pose, optionally writing to disk."""

    if isinstance(source, str):
        if not os.path.exists(source):
            logging.error("Input video file not found at: '%s'", source)
            return
        if os.path.getsize(source) < 1024:
            logging.error("Input video file at '%s' is too small to be a valid video.", source)
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Could not open video source '%s'.", source)
        return

    is_webcam = isinstance(source, int) and source == 0
    is_batch_mode = not is_webcam and output_path is not None

    logging.info("Running in batch mode..." if is_batch_mode else "Press 'q' to quit.")

    video_writer = None
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        logging.info("Saving output video to: %s", output_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            logging.error("Could not open video writer for path: %s", output_path)
            video_writer = None

    pbar = None
    if is_batch_mode:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing video frames")

    tracker = None
    track_colors = {}
    if enable_tracking and pose_model is None:
        tracker = DeepSort(max_age=30)
        logging.info("DeepSORT tracking enabled.")

    processed_frame_count = 0
    frame_idx = -1

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        if not success:
            break

        if pbar:
            pbar.update(1)

        if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
            continue

        if max_frames > 0 and processed_frame_count >= max_frames:
            logging.info("Reached max-frames limit of %d. Stopping processing.", max_frames)
            break

        processed_frame_count += 1

        # Run detection
        det_results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = det_results[0].plot()

        # Optional pose overlay
        if pose_model is not None:
            pose_conf = pose_conf_threshold if pose_conf_threshold is not None else conf_threshold
            pose_results = pose_model.predict(frame, conf=pose_conf, verbose=False)
            annotated_frame = pose_results[0].plot(img=annotated_frame)

        # Optional tracking (skip when pose overlay is on)
        if tracker is not None:
            detections_for_tracker = []
            for box in det_results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                width, height = x2 - x1, y2 - y1
                class_name = model.names[int(cls)]
                detections_for_tracker.append(([x1, y1, width, height], conf, class_name))

            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                if track_id not in track_colors:
                    track_colors[track_id] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )
                color = track_colors[track_id]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                det_class = getattr(track, "det_class", "object")
                det_conf = getattr(track, "det_conf", None)
                if det_conf is not None:
                    try:
                        label = f"{det_class} {float(det_conf):.2f} | ID:{track_id}"
                    except Exception:  # pragma: no cover - defensive fallback
                        label = f"{det_class} | ID:{track_id}"
                else:
                    label = f"{det_class} | ID:{track_id}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        if video_writer:
            video_writer.write(annotated_frame)

        if not is_batch_mode:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if pbar:
        pbar.close()
        logging.info("Batch processing complete.")

    cap.release()
    if video_writer:
        video_writer.release()
        logging.info("Output video saved to: %s", output_path)
    cv2.destroyAllWindows()
    logging.info("Video processing finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on a video or webcam.")
    parser.add_argument("-m", "--model", required=True, help="Path to the trained YOLO model file (.pt).")
    parser.add_argument("-i", "--input", default="0", help="Video path or '0' for webcam.")
    parser.add_argument("-p", "--probability", type=float, default=25, help="Minimum confidence (0-100).")
    parser.add_argument("-o", "--output", help="Directory to save the output video.")
    parser.add_argument("--enable-tracking", action="store_true", help="Enable object tracking with DeepSORT.")
    parser.add_argument("--enable-pose", action="store_true", help="Overlay pose estimation in addition to detection.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = all).")
    parser.add_argument("--frame-skip", type=int, default=0, help="Number of frames to skip between inferences.")
    args = parser.parse_args()

    video_source = 0 if args.input == "0" else args.input
    logging.info("Input source: %s", "Webcam" if video_source == 0 else video_source)

    pose_model_path = POSE_MODEL_DEFAULT if args.enable_pose else None
    detector = VideoDetector(model_path=args.model, pose_model_path=pose_model_path)
    confidence = args.probability / 100.0

    output_path = None
    if args.output:
        output_path = detector._generate_output_path(str(video_source), args.output)

    detector.process_video_interactive(
        source=video_source,
        conf_threshold=confidence,
        output_path=output_path,
        enable_tracking=args.enable_tracking,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        enable_pose=args.enable_pose,
    )


if __name__ == "__main__":
    setup_logging()
    main()
