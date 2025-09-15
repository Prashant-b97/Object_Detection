import unittest
from unittest.mock import patch, MagicMock, call, ANY
import argparse
import os
import sys
from io import StringIO

# Import the new core components for mocking
from detector.core import Detection, BoundingBox

# We need to import the module to be tested
import imagedetection

class TestYoloV8ImageDetection(unittest.TestCase):

    @patch('imagedetection.cv2.imread')
    @patch('imagedetection.cv2.imwrite')
    @patch('imagedetection.os.makedirs')
    @patch('imagedetection.print_detections')
    @patch('imagedetection.draw_detections')
    @patch('imagedetection.os.path.exists')
    @patch('imagedetection.ObjectDetector')
    def test_detect_objects_success(self, mock_ObjectDetector, mock_exists, mock_draw_detections, mock_print_detections, mock_makedirs, mock_imwrite, mock_imread):
        """Test the detect_objects function for successful detection."""
        mock_exists.return_value = True
        # Mock the ObjectDetector and its return value
        mock_model_instance = MagicMock()
        mock_detections = [
            Detection(class_name='person', confidence=0.95, box=BoundingBox(10, 20, 30, 40))
        ]
        mock_model_instance.detect_from_image.return_value = mock_detections
        mock_ObjectDetector.return_value = mock_model_instance

        # Prepare arguments
        args = argparse.Namespace(
            model='fake_model.pt',
            input='fake_image.jpg',
            output_dir='runs/detect',
            probability=25.0
        )

        # Run the function
        imagedetection.detect_objects(args)

        # Assertions
        mock_ObjectDetector.assert_called_once_with('fake_model.pt')
        mock_imread.assert_called_once_with('fake_image.jpg')
        mock_model_instance.detect_from_image.assert_called_once_with(ANY, conf_threshold=0.25)
        mock_draw_detections.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_imwrite.assert_called_once()
        mock_print_detections.assert_called_once()

    @patch('imagedetection.logging.error')
    @patch('imagedetection.os.path.exists')
    def test_detect_objects_input_file_not_found(self, mock_exists, mock_log_error):
        """Test detect_objects when the input file does not exist."""
        mock_exists.return_value = False

        args = argparse.Namespace(
            model='model.pt',
            input='nonexistent.jpg',
            # Add other necessary args for the function to run
            output_dir='runs/detect',
            probability=25.0
        )

        imagedetection.detect_objects(args)

        mock_log_error.assert_called_with("Input image not found at nonexistent.jpg")

    @patch('imagedetection.logging.info')
    def test_print_detections_no_objects(self, mock_log_info):
        """Test the print_detections function when no objects are found."""
        detections = []
        imagedetection.print_detections(detections, '/fake/dir')

        # Check that the relevant logging calls were made
        calls = mock_log_info.call_args_list
        self.assertIn(call('Output image saved in: /fake/dir'), calls)
        self.assertIn(call('Total objects detected: 0'), calls)
        self.assertIn(call('- No objects detected.'), calls)

    @patch('imagedetection.logging.info')
    @patch('imagedetection.YOLO')
    def test_train_model_success(self, mock_yolo, mock_log_info):
        """Test the train_model function."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = argparse.Namespace(
            data='dataset.yaml',
            pretrained_model='yolov8n.pt',
            epochs=10,
            batch_size=8
        )

        imagedetection.train_model(args)

        mock_yolo.assert_called_once_with('yolov8n.pt')
        mock_model_instance.train.assert_called_once_with(
            data='dataset.yaml',
            epochs=10,
            batch=8,
            imgsz=640,
            project='runs/train',
            name=ANY  # The name includes a timestamp, so we use ANY
        )
        mock_log_info.assert_any_call("Starting model training...")
        mock_log_info.assert_any_call("Training complete. The best model is saved in the 'runs/train/...' directory.")

    @patch('imagedetection.detect_objects')
    def test_main_detect_command(self, mock_detect_objects):
        """Test that main calls detect_objects for the 'detect' command."""
        test_args = ['imagedetection.py', 'detect', '--model', 'm.pt', '-i', 'i.jpg']
        with patch.object(sys, 'argv', test_args):
            imagedetection.main()
        
        mock_detect_objects.assert_called_once()
        call_args = mock_detect_objects.call_args[0][0]
        self.assertEqual(call_args.command, 'detect')
        self.assertEqual(call_args.model, 'm.pt')
        self.assertEqual(call_args.input, 'i.jpg')

    @patch('imagedetection.train_model')
    def test_main_train_command(self, mock_train):
        """Test that main calls train_model for the 'train' command."""
        test_args = ['imagedetection.py', 'train', '--data', 'd.yaml', '--epochs', '5']
        with patch.object(sys, 'argv', test_args):
            imagedetection.main()
        
        mock_train.assert_called_once()
        call_args = mock_train.call_args[0][0]
        self.assertEqual(call_args.command, 'train')
        self.assertEqual(call_args.data, 'd.yaml')
        self.assertEqual(call_args.epochs, 5)

    @patch('imagedetection.evaluate_model')
    def test_main_evaluate_command(self, mock_evaluate):
        """Test that main calls evaluate_model for the 'evaluate' command."""
        test_args = ['imagedetection.py', 'evaluate', '--data', 'd.yaml', '--model', 'latest']
        with patch.object(sys, 'argv', test_args):
            imagedetection.main()
        
        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args[0][0]
        self.assertEqual(call_args.command, 'evaluate')
        self.assertEqual(call_args.data, 'd.yaml')
        self.assertEqual(call_args.model, 'latest')

    @patch('imagedetection.os.path.isdir')
    @patch('imagedetection.os.listdir')
    @patch('imagedetection.os.path.exists')
    @patch('imagedetection.YOLO')
    def test_evaluate_model_latest(self, mock_yolo, mock_exists, mock_listdir, mock_isdir):
        """Test evaluate_model with the 'latest' keyword."""
        # --- Setup Mocks ---
        mock_isdir.return_value = True
        mock_listdir.return_value = ['train_20240101_000000', 'train_20240102_000000']
        mock_exists.return_value = True # The best.pt file exists
        
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = argparse.Namespace(data='coco8.yaml', model='latest')

        # --- Run Function ---
        imagedetection.evaluate_model(args)

        # --- Assertions ---
        # Check that it found the correct latest path
        expected_path = os.path.join('runs', 'train', 'train_20240102_000000', 'weights', 'best.pt')
        mock_yolo.assert_called_once_with(expected_path)
        mock_model_instance.val.assert_called_once_with(data='coco8.yaml')

    def test_main_no_command(self):
        """Test that main prints help and exits when no command is given."""
        test_args = ['imagedetection.py']
        with patch.object(sys, 'argv', test_args):
            # Check that sys.exit is called with code 1
            with self.assertRaises(SystemExit) as cm:
                # Capture stdout to check help message
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                     imagedetection.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("usage: imagedetection.py [-h] {detect,train,evaluate} ...", mock_stdout.getvalue())

if __name__ == "__main__":
    unittest.main()
