import argparse
import sys
import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def evaluate_tracking(model_path: str, video_path: str, max_frames: int = 200, conf: float = 0.25, min_persistence: int = 10):
    """
    Runs YOLOv8 + DeepSORT on the first `max_frames` frames and evaluates whether
    any track ID persists for at least `min_persistence` frames.

    Returns a dict with summary metrics.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if os.path.getsize(video_path) < 1024:
        raise ValueError(f"Input video seems too small: {video_path}")

    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    track_frames = defaultdict(int)  # track_id -> number of frames seen
    total_frames = 0

    try:
        while total_frames < max_frames and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, conf=conf, verbose=False)

            detections_for_tracker = []
            for box in results[0].boxes.data:
                x1, y1, x2, y2, score, cls = box.tolist()
                w, h = x2 - x1, y2 - y1
                class_name = model.names[int(cls)]
                detections_for_tracker.append(([x1, y1, w, h], score, class_name))

            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
            for t in tracks:
                if not t.is_confirmed():
                    continue
                track_frames[t.track_id] += 1

            total_frames += 1
    finally:
        cap.release()

    longest_persistence = max(track_frames.values()) if track_frames else 0

    return {
        "frames_processed": total_frames,
        "unique_tracks": len(track_frames),
        "longest_persistence": longest_persistence,
        "meets_threshold": longest_persistence >= min_persistence,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick test for DeepSORT tracking persistence.")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 .pt weights (e.g., yolov8n.pt)")
    parser.add_argument("--input", required=True, help="Path to a short test video")
    parser.add_argument("--frames", type=int, default=200, help="Max frames to process (default 200)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1), default 0.25")
    parser.add_argument("--min-persistence", type=int, default=10, help="Required frames a track must persist (default 10)")
    args = parser.parse_args()

    try:
        summary = evaluate_tracking(
            model_path=args.model,
            video_path=args.input,
            max_frames=args.frames,
            conf=args.conf,
            min_persistence=args.min_persistence,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print("Tracking test summary:")
    print(f"- Frames processed: {summary['frames_processed']}")
    print(f"- Unique tracks:    {summary['unique_tracks']}")
    print(f"- Longest persist:  {summary['longest_persistence']} frames")
    if summary["meets_threshold"]:
        print("RESULT: PASS — Tracking shows persistent IDs across frames.")
        sys.exit(0)
    else:
        print("RESULT: FAIL — No track persisted long enough.")
        sys.exit(1)


if __name__ == "__main__":
    main()

