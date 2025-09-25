import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.fastapi.api import app


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure you have a sample image for testing
        self.sample_image_path = Path("sample_data/Street Scene.png")
        if not self.sample_image_path.exists():
            raise FileNotFoundError(
                f"Test image not found at {self.sample_image_path}. "
                "Please ensure it exists."
            )

    def test_detect_endpoint(self):
        with open(self.sample_image_path, "rb") as img_file:
            response = self.client.post(
                "/detect", files={"file": ("image.png", img_file, "image/png")}
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        # Check if at least one object was detected (likely for this image)
        self.assertGreater(len(data), 0)
        # Check the structure of the first detection
        self.assertIn("box", data[0])
        self.assertIn("class_name", data[0])
        self.assertIn("confidence", data[0])

    def test_detect_endpoint_respects_conf_threshold(self):
        """Higher confidence thresholds should not increase detection count."""
        with open(self.sample_image_path, "rb") as img_file:
            default_response = self.client.post(
                "/detect", files={"file": ("image.png", img_file, "image/png")}
            )

        self.assertEqual(default_response.status_code, 200)
        default_data = default_response.json()

        with open(self.sample_image_path, "rb") as img_file:
            strict_response = self.client.post(
                "/detect?conf_threshold=0.9",
                files={"file": ("image.png", img_file, "image/png")},
            )

        self.assertEqual(strict_response.status_code, 200)
        strict_data = strict_response.json()

        if not default_data:
            self.skipTest("Baseline detection returned zero objects; cannot compare thresholds.")

        self.assertLess(
            len(strict_data),
            len(default_data),
            "Higher confidence threshold should yield fewer detections.",
        )
