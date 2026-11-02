# Real-Time Object Detection with MobileNet SSD

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%7C%205.x-green.svg)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, cross-platform computer vision application for real-time multi-class object detection. Supports **live webcams**, **video files**, **animated GIFs**, and **static images**.

This repository provides an optimized, modular, and dependency-light Python application [`detect.py`](file:///c:/Users/bless/GIT/Computer-Vision/detect.py) for real-time inference across multiple media formats.

---

## Architecture & Workflow

```
[Input Source]
  ├── Webcam (0, 1, ...)
  ├── Video (.mp4, .avi, .mov)
  ├── Animated GIF (.gif)
  └── Image (.jpg, .png)
        │
        ▼
[Frame Preprocessing]
  ├── Aspect-ratio preserving resize (e.g. 600px width)
  └── Format normalization (RGB uint8 / BGR blob)
        │
        ▼
[Inference Engine]
  ├── ONNX Runtime (Default, COCO 80+ classes, compatible with OpenCV 4.x & 5.x)
  └── OpenCV Caffe DNN (Legacy, Pascal VOC 20 classes, compatible with OpenCV <= 4.x)
        │
        ▼
[Post-Processing & Annotation]
  ├── Confidence threshold filtering (> 0.50 default)
  ├── Bounding box calculation & coordinate clamping
  ├── Color-coded class labels & probability overlay
  └── Live FPS tracking
        │
        ▼
[Output Destination]
  ├── Real-time GUI Window preview (with 'q' / Esc key to exit)
  └── Saved Media (.mp4 video, .gif animation, or .jpg image)
```

---

## Key Features

- **Multi-Source Ingestion**: Process live webcam feeds, video files, multi-frame animated GIFs, or individual photos through a single unified CLI.
- **Modernized Backend (OpenCV 5.x Ready)**: Built-in support for **ONNX Runtime** (`ssd_mobilenet_v1_10.onnx`), resolving the deprecation of Caffe importers in modern OpenCV 5.0+ while remaining backward-compatible with legacy Caffe models.
- **Automated Model Provisioning**: Weights and configuration files are automatically verified and fetched from verified mirrors on first execution if missing.
- **Headless-Aware Execution**: Automatically detects headless environments (`opencv-python-headless`) and falls back safely to file export without raising GUI display exceptions.
- **Export Pipeline**: Directly export annotated results to high-quality animated GIFs (`.gif`), video files (`.mp4`), or images (`.jpg`).
- **Zero Heavyweight Wrappers**: Implemented with native NumPy, OpenCV, and Pillow, avoiding unnecessary dependencies.

---

## Project Structure

```
Computer-Vision/
├── detect.py                           # Core detection script and CLI application
├── real_time.gif                       # Sample animated GIF input
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git exclusions for models and output files
├── README.md                           # Documentation
└── models/
    ├── ssd_mobilenet_v1_10.onnx        # ONNX SSD-MobileNet model (COCO classes)
    ├── MobileNetSSD_deploy.caffemodel  # Caffe pre-trained weights (Pascal VOC)
    └── MobileNetSSD_deploy.prototxt.txt# Caffe network architecture definition
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Computer-Vision.git
cd Computer-Vision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Display Mode Note:** If you are using `opencv-python-headless`, desktop popup windows (`cv2.imshow`) are disabled by design. To enable interactive GUI preview windows:
> ```bash
> pip uninstall opencv-python-headless -y
> pip install opencv-python
> ```

---

## Usage Guide

### 1. Detect Objects in an Animated GIF
Process all frames of a GIF and generate an annotated output GIF:
```bash
python detect.py -i real_time.gif -o output_realtime.gif
```

### 2. Live Webcam Stream
Run real-time object detection using your default webcam (index `0`):
```bash
python detect.py -i 0
```
*(Press **`q`** or **`ESC`** in the window to quit)*

### 3. Video File Processing
Analyze a video file and save the labeled result to an MP4 video:
```bash
python detect.py -i footage.mp4 -o output_footage.mp4
```

### 4. Static Image Detection
Annotate a single photo:
```bash
python detect.py -i street.jpg -o annotated_street.jpg
```

### 5. Custom Confidence Threshold & Frame Resizing
Filter out weaker predictions (e.g., minimum 65% confidence) and scale frames to width 800px:
```bash
python detect.py -i real_time.gif --confidence 0.65 --width 800 -o output.gif
```

### 6. Run Headless (No GUI Window)
Ideal for background servers, Docker containers, or batch jobs:
```bash
python detect.py -i real_time.gif -o output.gif --no-display
```

### 7. Legacy Caffe Model (OpenCV $\le 4.x$)
To use the original Pascal VOC Caffe weights:
```bash
python detect.py -i real_time.gif -m models/MobileNetSSD_deploy.caffemodel -p models/MobileNetSSD_deploy.prototxt.txt
```

---

## Command-Line Arguments Reference

| Option | Flag | Type | Default | Description |
|---|:---:|:---:|---|---|
| `--input` | `-i` | `str` | `"0"` | Path to input (`.gif`, `.mp4`, `.jpg`, `.png`) or camera index (`0`) |
| `--model` | `-m` | `str` | `models/ssd_mobilenet_v1_10.onnx` | Path to `.onnx` or `.caffemodel` weights |
| `--prototxt` | `-p` | `str` | `models/MobileNetSSD_deploy.prototxt.txt` | Path to Caffe deploy prototxt file (Caffe only) |
| `--confidence`| `-c` | `float`| `0.5` | Minimum confidence threshold (`0.0` to `1.0`) |
| `--width` | `-w` | `int` | `600` | Target resize width in pixels (`0` preserves original size) |
| `--output` | `-o` | `str` | `None` | Output file path (`.gif`, `.mp4`, `.avi`, `.jpg`, `.png`) |
| `--no-display`| | `flag` | `False` | Run in headless mode without opening GUI windows |

---

## Supported Object Categories

### ONNX Model (COCO Dataset - 80 Classes)
`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`.

### Caffe Model (Pascal VOC Dataset - 20 Classes)
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

---

## Troubleshooting

- **`cv2.error: The function is not implemented`**:
  You are running `opencv-python-headless`. If you wish to view live windows, replace it with `pip install opencv-python`. Alternatively, pass `--output <filename>` to save results directly.
- **`cv2.dnn has no attribute readNetFromCaffe`**:
  OpenCV 5.0 removed the Caffe importer. Run the default ONNX model (`python detect.py`) which works seamlessly on all OpenCV versions via ONNX Runtime.
- **Webcam Access Issues**:
  Ensure your operating system allows camera access for your terminal/Python application, and verify the camera index (try `-i 0` or `-i 1`).
