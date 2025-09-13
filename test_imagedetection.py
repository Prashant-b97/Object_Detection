import unittest
from unittest.mock import patch, MagicMock, call, ANY
import argparse
import os
import sys
from io import StringIO

# We need to import the module to be tested
import imagedetection

class TestYoloV8ImageDetection(unittest.TestCase):

    @patch('imagedetection.os.path.exists')
    @patch('imagedetection.YOLO')
    def test_detect_objects_success(self, mock_yolo, mock_exists):
        """Test the detect_objects function for successful detection."""
        mock_exists.return_value = True
        
        # Mock the YOLO model and its predict method
        mock_model_instance = MagicMock()
        # Create a more realistic mock results object
        mock_predict_result = MagicMock()
        mock_predict_result.save_dir = '/fake/run/detect'
        box1 = MagicMock()
        box1.cls, box1.conf = [0], [0.95]
        box2 = MagicMock()
        box2.cls, box2.conf = [1], [0.80]
        mock_predict_result.boxes = [box1, box2]
        mock_predict_result.names = {0: 'person', 1: 'car'}
        mock_model_instance.predict.return_value = [mock_predict_result]
        mock_yolo.return_value = mock_model_instance

        # Prepare arguments
        args = argparse.Namespace(
            model='fake_model.pt',
            input='fake_image.jpg',
            output_dir='runs/detect',
            probability=25.0
        )

        # Capture print output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Run the function
        imagedetection.detect_objects(args)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Assertions
        mock_yolo.assert_called_once_with('fake_model.pt')
        mock_model_instance.predict.assert_called_once_with(
            source='fake_image.jpg',
            save=True,
            conf=0.25,
            project='runs/detect',
            exist_ok=True
        )
        output = captured_output.getvalue()
        self.assertIn("Loading YOLOv8 model and starting object detection...", output)
        self.assertIn("Detection complete.", output)
        self.assertIn("Output image saved in: /fake/run/detect", output)
        self.assertIn("Total objects detected: 2", output)
        self.assertIn("- person (95.00%)", output)
        self.assertIn("- car (80.00%)", output)

    @patch('imagedetection.os.path.exists')
    def test_detect_objects_input_file_not_found(self, mock_exists):
        """Test detect_objects when the input file does not exist."""
        mock_exists.side_effect = [False, True]  # Input image doesn't exist

        args = argparse.Namespace(model='model.pt', input='non_existent_image.jpg', output_dir='runs/detect', probability=25.0)

        captured_stderr = StringIO()
        sys.stderr = captured_stderr
        imagedetection.detect_objects(args)
        sys.stderr = sys.__stderr__

        self.assertIn("Error: Input image not found", captured_stderr.getvalue())

    @patch('imagedetection.os.path.exists')
    def test_detect_objects_model_file_not_found(self, mock_exists):
        """Test detect_objects when the model file does not exist."""
        mock_exists.side_effect = [True, False]  # Model file doesn't exist

        args = argparse.Namespace(model='model.pt', input='image.jpg')

        captured_stderr = StringIO()
        sys.stderr = captured_stderr
        imagedetection.detect_objects(args)
        sys.stderr = sys.__stderr__

        self.assertIn("Error: Model file not found", captured_stderr.getvalue())

    def test_print_detections_no_objects(self):
        """Test the print_detections function when no objects are found."""
        mock_result = MagicMock()
        mock_result.save_dir = '/fake/run/detect'
        mock_result.boxes = []
        mock_results_list = [mock_result]

        captured_output = StringIO()
        sys.stdout = captured_output
        imagedetection.print_detections(mock_results_list)
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("Total objects detected: 0", output)
        self.assertIn("- No objects detected.", output)

    @patch('imagedetection.YOLO')
    def test_train_model_success(self, mock_yolo):
        """Test the train_model function."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        args = argparse.Namespace(
            data='dataset.yaml',
            pretrained_model='yolov8n.pt',
            epochs=10,
            batch_size=8
        )

        captured_output = StringIO()
        sys.stdout = captured_output
        imagedetection.train_model(args)
        sys.stdout = sys.__stdout__

        mock_yolo.assert_called_once_with('yolov8n.pt')
        mock_model_instance.train.assert_called_once_with(
            data='dataset.yaml',
            epochs=10,
            batch=8,
            imgsz=640,
            project='runs/train',
            name=ANY  # The name includes a timestamp, so we use ANY
        )
        self.assertIn("Starting model training...", captured_output.getvalue())
        self.assertIn("Training complete.", captured_output.getvalue())

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
    def test_main_train_command(self, mock_train_model):
        """Test that main calls train_model for the 'train' command."""
        test_args = ['imagedetection.py', 'train', '--data', 'd.yaml', '--epochs', '5']
        with patch.object(sys, 'argv', test_args):
            imagedetection.main()
        
        mock_train_model.assert_called_once()
        call_args = mock_train_model.call_args[0][0]
        self.assertEqual(call_args.command, 'train')
        self.assertEqual(call_args.data, 'd.yaml')
        self.assertEqual(call_args.epochs, 5)

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
        self.assertIn("usage: imagedetection.py [-h] {detect,train} ...", mock_stdout.getvalue())

if __name__ == "__main__":
    unittest.main()
