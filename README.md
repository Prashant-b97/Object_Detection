# ObjectDetection
🚀 Train custom YOLOv8 models and detect objects with this powerful Python CLI. Features easy-to-use commands for both transfer learning and inference.

# YOLOv8 Object Detection and Training Tool

This project provides a command-line tool for training custom YOLOv8 object detection models and running inference on images. It is built using the `ultralytics` Python library.


This project provides a command-line tool for training custom YOLOv8 object detection models and running inference on images. It is built using the `ultralytics` Python library.

## Sample Detection

Below is an example of running the detector on a sample image.
| Input Image | Output Detection |
![alt text](<ChatGPT Image Sep 9, 2025, 10_27_21 PM.png>)|![alt text](<runs/detect/predict/ChatGPT Image Sep 9, 2025, 10_27_21 PM.jpg>)|
| !Sample Input | !Sample Output |


---

## Features

- **Train**: Train a YOLOv8 model on a custom dataset in YOLO format.
- **Detect**: Perform object detection on a single image using a pretrained or custom-trained model.
- **Easy-to-use CLI**: Simple and clear commands for both training and detection.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Prashant-b97/Object_Detection.git
    cd Object_Detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The script has two main commands: `train` and `detect`.

### Training a Custom Model

To train a model, you need a dataset YAML file (like `coco8.yaml`).
 
```bash
python imagedetection.py train \
    --data path/to/your_dataset.yaml \
    --epochs 50 \
    --batch-size 8
```

The best trained model will be saved in the `runs/train/.../weights/` directory as `best.pt`.

### Detecting Objects in an Image

Use the `detect` command to run inference on an image. You must provide a path to a model file (`.pt`).

**Example using a pretrained model:**
*(The script will auto-download `yolov8n.pt` if not present)*
```bash
python imagedetection.py detect \
    --input sample_input/image.jpg \
    --model yolov8n.pt
```

**Example using your custom-trained model:**
```bash
python imagedetection.py detect \
    --input sample_input/image.jpg \
    --model runs/train/your_experiment/weights/best.pt
```

The output image will be saved in the `runs/detect/predict/` directory.

---

## Running Tests

To ensure the script is working correctly, you can run the included unit tests:

```bash
python -m unittest test_imagedetection.py
```

---

## License

This project is licensed under the AGPL-3.0 License. See the `LICENSE` file for details. This is required due to the project's use of the `ultralytics` library.
```
