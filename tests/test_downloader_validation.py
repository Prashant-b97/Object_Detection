import os
import io
import tempfile
import unittest
from unittest.mock import patch

# Add project root to the Python path to allow imports from utils
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.download_video as dv


class MockResponse:
    """A mock for the requests.Response object."""
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = headers or {"content-length": str(len(body))}

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        bio = io.BytesIO(self._body)
        while True:
            chunk = bio.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeCapOK:
    """A mock for a successful cv2.VideoCapture."""
    def __init__(self, path):
        pass

    def isOpened(self):
        return True

    def read(self):
        return True, b"frame_data"

    def release(self):
        pass


class FakeCapFailOpen:
    """A mock for a failing cv2.VideoCapture."""
    def __init__(self, path):
        pass

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


class DownloaderValidationTests(unittest.TestCase):
    def test_download_validation_success(self):
        """Test that a valid 'download' succeeds and the file is kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "ok.mp4")
            with patch("utils.download_video.requests.get", return_value=MockResponse(b"video")):
                with patch("utils.download_video.cv2.VideoCapture", side_effect=FakeCapOK):
                    ok = dv.download_file("http://example.com/video.mp4", out_path)
            self.assertTrue(ok, "download_file should return True on success")
            self.assertTrue(os.path.exists(out_path), "File should exist after successful validation")

    def test_download_validation_failure_removes_file(self):
        """Test that an invalid 'download' fails and the file is removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "bad.mp4")
            with patch("utils.download_video.requests.get", return_value=MockResponse(b"not-a-video")):
                with patch("utils.download_video.cv2.VideoCapture", side_effect=FakeCapFailOpen):
                    ok = dv.download_file("http://example.com/video.mp4", out_path)
            self.assertFalse(ok, "download_file should return False on failure")
            self.assertFalse(os.path.exists(out_path), "Invalid file should be removed after failed validation")