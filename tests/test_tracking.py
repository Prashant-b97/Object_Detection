import argparse
import sys
import os
from collections import defaultdict
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


def _load_yolo(model_path: str):
    """Lazily import and instantiate the YOLO model."""
    from ultralytics import YOLO  # Deferred import to keep tests lightweight.

    return YOLO(model_path)


def _create_tracker(max_age: int = 30):
    """Lazily import and instantiate the DeepSORT tracker."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    return DeepSort(max_age=max_age)


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

    model = _load_yolo(model_path)
    tracker = _create_tracker(max_age=30)

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


class TrackingEvaluationTests(unittest.TestCase):
    def test_persistent_track_counts_toward_threshold(self):
        """Ensure evaluate_tracking flags persistent DeepSORT tracks as passing."""
        fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        fake_box_array = np.array([[0, 0, 1, 1, 0.9, 0]], dtype=float)

        fake_track = MagicMock()
        fake_track.is_confirmed.return_value = True
        fake_track.track_id = 42

        with patch('tests.test_tracking.os.path.exists', return_value=True), \
             patch('tests.test_tracking.os.path.getsize', return_value=2048), \
             patch('tests.test_tracking.cv2.VideoCapture') as mock_videocapture, \
             patch('tests.test_tracking._load_yolo') as mock_load_yolo, \
             patch('tests.test_tracking._create_tracker') as mock_create_tracker:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = [(True, fake_frame)] * 5 + [(False, None)]
            mock_videocapture.return_value = mock_cap

            mock_model = MagicMock()
            mock_model.names = {0: 'person'}
            mock_result = MagicMock()
            mock_result.boxes.data = fake_box_array
            mock_model.predict.return_value = [mock_result]
            mock_load_yolo.return_value = mock_model

            mock_tracker = MagicMock()
            mock_tracker.update_tracks.side_effect = [[fake_track]] * 5
            mock_create_tracker.return_value = mock_tracker

            summary = evaluate_tracking(
                model_path='weights.pt',
                video_path='video.mp4',
                max_frames=5,
                conf=0.25,
                min_persistence=3,
            )

        self.assertEqual(summary['frames_processed'], 5)
        self.assertEqual(summary['unique_tracks'], 1)
        self.assertEqual(summary['longest_persistence'], 5)
        self.assertTrue(summary['meets_threshold'])
        mock_tracker.update_tracks.assert_called()
        mock_model.predict.assert_called()
        mock_cap.release.assert_called_once()


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
