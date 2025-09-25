import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is on the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector.core import ObjectDetector


class TestCoreDetector(unittest.TestCase):
    """Integration tests for detector core utilities."""

    @classmethod
    def setUpClass(cls):
        cls.detector = ObjectDetector(model_path="yolov8n.pt")
        sample_path = Path("sample_data/Street Scene.png")
        if not sample_path.exists():
            raise FileNotFoundError(f"Test image not found at {sample_path}")
        image_bgr = cv2.imread(str(sample_path))
        cls.sample_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def test_detect_with_heatmap_output_shape(self):
        """detect_with_heatmap should return an RGB image matching the input dimensions."""
        output_image = self.detector.detect_with_heatmap(self.sample_image_rgb)

        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape[:2], self.sample_image_rgb.shape[:2])
        self.assertEqual(output_image.ndim, 3)
        self.assertEqual(output_image.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
