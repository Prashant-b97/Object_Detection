import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
from io import StringIO
import numpy as np

# We need to import the module to be tested
import videodetection

class TestVideoDetection(unittest.TestCase):

    @patch('videodetection.cv2.destroyAllWindows')
    @patch('videodetection.cv2.waitKey')
    @patch('videodetection.cv2.imshow')
    @patch('videodetection.cv2.VideoCapture')
    def test_process_video_success_and_end_of_stream(self, mock_videocapture, mock_imshow, mock_waitkey, mock_destroy):
        """Test successful video processing until the stream ends."""
        # --- Setup Mocks ---
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame), (True, mock_frame), (False, None)]
        mock_videocapture.return_value = mock_cap

        # Mock YOLO model and its results
        mock_yolo_model = MagicMock()
        mock_result = MagicMock()
        mock_annotated_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.plot.return_value = mock_annotated_frame
        mock_yolo_model.predict.return_value = [mock_result]

        # Mock waitKey to not quit
        mock_waitkey.return_value = -1

        # --- Run Function ---
        videodetection.process_video(mock_yolo_model, 'fake.mp4', 0.5)

        # --- Assertions ---
        self.assertEqual(mock_cap.read.call_count, 3, "Should read until stream ends")
        self.assertEqual(mock_yolo_model.predict.call_count, 2, "Should predict on two good frames")
        self.assertEqual(mock_imshow.call_count, 2, "Should display two good frames")
        mock_imshow.assert_called_with("YOLOv8 Inference", mock_annotated_frame)
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('videodetection.cv2.destroyAllWindows')
    @patch('videodetection.cv2.waitKey')
    @patch('videodetection.cv2.imshow')
    @patch('videodetection.cv2.VideoCapture')
    def test_process_video_quit_with_q(self, mock_videocapture, mock_imshow, mock_waitkey, mock_destroy):
        """Test video processing and quitting with the 'q' key."""
        # --- Setup Mocks ---
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_videocapture.return_value = mock_cap

        mock_yolo_model = MagicMock()
        mock_yolo_model.predict.return_value = [MagicMock(plot=lambda: mock_frame)]

        # Simulate pressing 'q' on the second frame check
        mock_waitkey.side_effect = [-1, ord('q')]

        # --- Run Function ---
        videodetection.process_video(mock_yolo_model, 0, 0.5) # Use webcam source '0' for interactive mode

        # --- Assertions ---
        self.assertEqual(mock_cap.read.call_count, 2, "Should read two frames before quitting")
        self.assertEqual(mock_yolo_model.predict.call_count, 2, "Should predict on two frames")
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('videodetection.cv2.VideoCapture')
    def test_process_video_source_not_opened(self, mock_videocapture):
        """Test the case where the video source cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        captured_output = StringIO()
        sys.stdout = captured_output

        videodetection.process_video(MagicMock(), 'nonexistent.mp4', 0.5)

        sys.stdout = sys.__stdout__

        self.assertIn("Error: Could not open video source", captured_output.getvalue())
        mock_cap.release.assert_not_called()

    @patch('videodetection.process_video')
    @patch('videodetection.YOLO')
    def test_main_webcam_input(self, mock_YOLO, mock_process_video):
        """Test main function with default webcam input."""
        mock_model_instance = MagicMock()
        mock_YOLO.return_value = mock_model_instance
        
        test_args = ['videodetection.py', '--model', 'm.pt']
        with patch.object(sys, 'argv', test_args):
            videodetection.main()

        mock_YOLO.assert_called_once_with('m.pt')
        mock_process_video.assert_called_once_with(mock_model_instance, 0, 0.25, output_path=ANY)

    @patch('videodetection.os.path.join')
    @patch('videodetection.os.path.splitext')
    @patch('videodetection.os.path.basename')
    @patch('videodetection.process_video')
    @patch('videodetection.YOLO')
    def test_main_video_file_input_with_output(self, mock_YOLO, mock_process_video, mock_basename, mock_splitext, mock_join):
        """Test main function with a video file path and output directory."""
        mock_model_instance = MagicMock()
        mock_YOLO.return_value = mock_model_instance
        mock_basename.return_value = 'video.mp4'
        mock_splitext.return_value = ('video', '.mp4')
        mock_join.return_value = 'output/video_timestamp.mp4'

        test_args = ['videodetection.py', '--model', 'm.pt', '--input', 'video.mp4', '-p', '50', '-o', 'output']
        with patch.object(sys, 'argv', test_args):
            videodetection.main()

        mock_YOLO.assert_called_once_with('m.pt')
        mock_process_video.assert_called_once_with(mock_model_instance, 'video.mp4', 0.5, output_path='output/video_timestamp.mp4')

    @patch('videodetection.tqdm')
    @patch('videodetection.cv2.VideoWriter')
    @patch('videodetection.cv2.VideoCapture')
    @patch('videodetection.os.makedirs')
    def test_process_video_batch_mode(self, mock_makedirs, mock_videocapture, mock_writer, mock_tqdm):
        """Test video processing in non-interactive batch mode."""
        # --- Setup Mocks ---
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [1920, 1080, 30, 10]  # width, height, fps, frame_count
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame)] * 10 + [(False, None)]
        mock_videocapture.return_value = mock_cap

        mock_writer_instance = MagicMock()
        mock_writer_instance.isOpened.return_value = True
        mock_writer.return_value = mock_writer_instance

        mock_yolo_model = MagicMock()
        mock_result = MagicMock()
        mock_result.plot.return_value = mock_frame
        mock_yolo_model.predict.return_value = [mock_result]

        # --- Run Function ---
        videodetection.process_video(mock_yolo_model, 'video.mp4', 0.5, output_path='output/vid.mp4')

        # --- Assertions ---
        mock_makedirs.assert_called_once_with('output', exist_ok=True)
        self.assertEqual(mock_cap.read.call_count, 11)  # 10 good frames, 1 end
        self.assertEqual(mock_yolo_model.predict.call_count, 10)
        self.assertEqual(mock_writer_instance.write.call_count, 10)
        mock_writer_instance.release.assert_called_once()

if __name__ == "__main__":
    unittest.main()
