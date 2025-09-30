"""Benchmark YOLO inference speed across resolutions.

Example usage:
    python scripts/benchmark_fps.py \
        --model runs/train/train_20250928_223128/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


def parse_resolutions(res_string: str) -> List[int]:
    values = []
    for part in res_string.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid resolution value: {part}") from exc
    if not values:
        raise argparse.ArgumentTypeError("Resolution list cannot be empty")
    return values


def benchmark(model: YOLO, image, resolution: int, runs: int) -> float:
    resized = cv2.resize(image, (resolution, resolution))
    model.predict(resized, imgsz=resolution, verbose=False, save=False)  # warmup
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(resized, imgsz=resolution, verbose=False, save=False)
        timings.append(time.perf_counter() - start)
    avg_time = sum(timings) / len(timings)
    return avg_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FPS vs. image resolution for a YOLO model")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO weights (.pt)")
    parser.add_argument(
        "--image",
        default="sample_data/Street Scene.jpg",
        help="Reference image used for inference",
    )
    parser.add_argument(
        "--resolutions",
        default="320,480,640,800,960",
        help="Comma-separated list of square image sizes to benchmark",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per resolution")
    parser.add_argument("--output", default="reports/fps_metrics.json", help="Path to write the FPS metrics JSON")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found at {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        parser.error(f"Failed to read image at {image_path}")

    try:
        resolutions = parse_resolutions(args.resolutions)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model weights not found at {model_path}")

    model = YOLO(str(model_path))

    measurements = []
    for res in resolutions:
        avg_time = benchmark(model, image, res, args.runs)
        fps = (1.0 / avg_time) if avg_time > 0 else 0.0
        measurements.append(
            {
                "resolution": res,
                "avg_time_ms": avg_time * 1000.0,
                "fps": fps,
                "runs": args.runs,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model_path),
        "image": str(image_path),
        "runs_per_resolution": args.runs,
        "measurements": measurements,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Benchmark complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
