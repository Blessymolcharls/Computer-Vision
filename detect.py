#!/usr/bin/env python3
"""
Object Detection using MobileNet SSD (ONNX & Caffe support).
Supports local execution for webcam, video, GIF, and image feeds.

Backends:
- ONNX Runtime / OpenCV DNN (SSD MobileNet v1, COCO 80/90 classes) [Default, compatible with OpenCV 5.x & 4.x]
- Caffe (MobileNet SSD, Pascal VOC 20 classes) [Compatible with OpenCV <= 4.x]
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

# Try importing onnxruntime (preferred for modern OpenCV 5.x compatibility)
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Pascal VOC 20 classes (+ background) for Caffe MobileNet SSD
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# COCO 90 labels dictionary for SSD MobileNet ONNX
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

# Deterministic colors for bounding boxes
np.random.seed(42)
VOC_COLORS = np.random.uniform(0, 255, size=(len(VOC_CLASSES), 3)).astype("uint8")
COCO_COLORS = np.random.uniform(0, 255, size=(100, 3)).astype("uint8")

DOWNLOAD_URLS = {
    "ssd_mobilenet_v1_10.onnx": "https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_10/resolve/main/ssd_mobilenet_v1_10.onnx",
    "MobileNetSSD_deploy.prototxt.txt": "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://huggingface.co/spaces/Imran606/cds/resolve/main/MobileNetSSD_deploy.caffemodel"
}


class FPSCounter:
    """Lightweight FPS tracker without imutils dependency."""
    def __init__(self):
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self):
        self._end = time.perf_counter()

    def update(self):
        self._num_frames += 1

    def elapsed(self):
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start if self._start else 0.0

    def fps(self):
        elapsed = self.elapsed()
        return self._num_frames / elapsed if elapsed > 0 else 0.0


def download_file(url: str, dest_path: str):
    """Download a file with visual progress."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    filename = os.path.basename(dest_path)
    print(f"[INFO] Downloading {filename}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                mb_cur = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {pct:5.1f}% ({mb_cur:.1f}MB / {mb_tot:.1f}MB)")
                sys.stdout.flush()

    print(f"\n[INFO] Saved {filename} to {dest_path}")


def ensure_file(filepath: str, download_key: str):
    """Check if file exists, download if missing and recognized."""
    if not os.path.exists(filepath):
        if download_key in DOWNLOAD_URLS:
            download_file(DOWNLOAD_URLS[download_key], filepath)
        else:
            raise FileNotFoundError(f"Required model file not found: {filepath}")


class ObjectDetector:
    """Wrapper supporting both ONNX Runtime and OpenCV Caffe inference."""
    def __init__(self, model_path: str, prototxt_path: str = None, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.model_path = model_path
        self.is_onnx = model_path.lower().endswith(".onnx")

        if self.is_onnx:
            ensure_file(model_path, "ssd_mobilenet_v1_10.onnx")
            if not HAS_ORT:
                raise ImportError(
                    "onnxruntime is required for .onnx models. Install with: pip install onnxruntime"
                )
            print(f"[INFO] Initializing ONNX Runtime session with {model_path}...")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            # Caffe model
            ensure_file(model_path, "MobileNetSSD_deploy.caffemodel")
            if prototxt_path:
                ensure_file(prototxt_path, "MobileNetSSD_deploy.prototxt.txt")
            else:
                raise ValueError("Caffe model requires --prototxt path.")

            if not hasattr(cv2.dnn, "readNetFromCaffe"):
                raise RuntimeError(
                    f"OpenCV {cv2.__version__} does not support Caffe (cv2.dnn.readNetFromCaffe removed).\n"
                    f"Please run with the ONNX model instead: python detect.py --model models/ssd_mobilenet_v1_10.onnx"
                )
            print(f"[INFO] Loading Caffe network ({prototxt_path}, {model_path})...")
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect(self, frame: np.ndarray):
        """
        Runs object detection on a single BGR frame.
        Returns: (annotated_frame, list_of_detections)
        """
        (h, w) = frame.shape[:2]
        found_objects = []

        if self.is_onnx:
            # ONNX SSD-MobileNet expects uint8 RGB image tensor (1, 300, 300, 3)
            resized = cv2.resize(frame, (300, 300))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0)

            boxes, classes, scores, num = self.session.run(
                None, {self.input_name: input_tensor}
            )

            num_detections = int(num[0])
            for i in range(num_detections):
                score = float(scores[0][i])
                if score >= self.min_confidence:
                    cls_id = int(classes[0][i])
                    class_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")

                    # Box format: [ymin, xmin, ymax, xmax] normalized [0, 1]
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    startX = int(xmin * w)
                    startY = int(ymin * h)
                    endX = int(xmax * w)
                    endY = int(ymax * h)

                    # Clamp boundaries
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w - 1, endX)
                    endY = min(h - 1, endY)

                    color = [int(c) for c in COCO_COLORS[cls_id % len(COCO_COLORS)]]

                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    label = f"{class_name}: {score * 100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 15
                    cv2.putText(
                        frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                    found_objects.append({
                        "class": class_name,
                        "confidence": score,
                        "box": (startX, startY, endX, endY)
                    })
        else:
            # Caffe MobileNet SSD expects 300x300 float32 blob with scale 0.007843 and mean 127.5
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                scalefactor=0.007843,
                size=(300, 300),
                mean=127.5
            )
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence >= self.min_confidence:
                    idx = int(detections[0, 0, i, 1])
                    class_name = VOC_CLASSES[idx]

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w - 1, endX)
                    endY = min(h - 1, endY)

                    color = [int(c) for c in VOC_COLORS[idx]]

                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    label = f"{class_name}: {confidence * 100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 15
                    cv2.putText(
                        frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                    found_objects.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": (startX, startY, endX, endY)
                    })

        return frame, found_objects


def resize_frame(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Aspect-ratio-preserving resize."""
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def show_frame_safe(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> tuple[bool, int]:
    """
    Safely calls cv2.imshow with automatic fallback if headless OpenCV is active.
    Returns (gui_available, key_pressed).
    """
    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        return True, key
    except cv2.error:
        return False, -1


def process_gif(gif_path, detector, target_width, output_path, display):
    """Process an animated GIF frame by frame."""
    print(f"[INFO] Opening GIF: {gif_path}")
    gif = Image.open(gif_path)
    fps_tracker = FPSCounter().start()
    out_frames_pil = []
    gui_supported = True

    try:
        for idx, frame_pil in enumerate(ImageSequence.Iterator(gif)):
            rgb_frame = np.array(frame_pil.convert("RGB"))
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if target_width:
                bgr_frame = resize_frame(bgr_frame, width=target_width)

            annotated, found = detector.detect(bgr_frame)
            fps_tracker.update()

            if output_path and output_path.lower().endswith(".gif"):
                rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                out_frames_pil.append(Image.fromarray(rgb_annotated))

            if display and gui_supported:
                gui_supported, key = show_frame_safe("Object Detection", annotated, wait_ms=40)
                if not gui_supported:
                    print("[NOTICE] Headless OpenCV detected; desktop GUI display is disabled.")
                    print("         Use --output <path.gif> to save results, or install GUI OpenCV.")
                elif key == ord("q") or key == 27:
                    print("[INFO] Quitting loop.")
                    break
    finally:
        fps_tracker.stop()

    if output_path and out_frames_pil:
        print(f"[INFO] Saving annotated GIF to {output_path}...")
        duration = gif.info.get("duration", 100)
        out_frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=out_frames_pil[1:],
            loop=0,
            duration=duration
        )
        print(f"[INFO] Successfully saved to {output_path}")

    return fps_tracker


def process_video_or_camera(src, detector, target_width, output_path, display):
    """Process a live camera feed or video file."""
    is_cam = False
    try:
        cam_idx = int(src)
        cap = cv2.VideoCapture(cam_idx)
        is_cam = True
        print(f"[INFO] Connecting to camera index {cam_idx}...")
    except ValueError:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Input file not found: {src}")
        cap = cv2.VideoCapture(src)
        print(f"[INFO] Reading video file: {src}")

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    fps_tracker = FPSCounter().start()
    writer = None
    gui_supported = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if target_width:
                frame = resize_frame(frame, width=target_width)

            annotated, found = detector.detect(frame)
            fps_tracker.update()

            # Initialize video writer if output is requested
            if output_path and writer is None:
                (h, w) = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_val = cap.get(cv2.CAP_PROP_FPS)
                if not fps_val or fps_val < 1:
                    fps_val = 30.0
                writer = cv2.VideoWriter(output_path, fourcc, fps_val, (w, h), True)

            if writer is not None:
                writer.write(annotated)

            if display and gui_supported:
                live_fps = fps_tracker.fps()
                cv2.putText(
                    annotated, f"FPS: {live_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                gui_supported, key = show_frame_safe(
                    "Object Detection", annotated, wait_ms=1 if is_cam else 25
                )
                if not gui_supported:
                    print("[NOTICE] Headless OpenCV detected; desktop GUI display is disabled.")
                    print("         Use --output <path.mp4> to save results.")
                elif key == ord("q") or key == 27:
                    print("[INFO] Quitting stream.")
                    break
    finally:
        fps_tracker.stop()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved to {output_path}")

    return fps_tracker


def process_image(image_path, detector, target_width, output_path, display):
    """Process a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image at {image_path}")

    if target_width:
        image = resize_frame(image, width=target_width)

    start = time.perf_counter()
    annotated, objects = detector.detect(image)
    elapsed = time.perf_counter() - start

    print(f"[INFO] Detection finished in {elapsed:.3f}s. Objects found: {len(objects)}")
    for obj in objects:
        print(f"  - {obj['class']} ({obj['confidence'] * 100:.1f}%) at {obj['box']}")

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved output image to {output_path}")

    if display:
        gui_supported, _ = show_frame_safe("Object Detection", annotated, wait_ms=0)
        if not gui_supported:
            print("[NOTICE] Headless OpenCV detected; image cannot be displayed in a window.")
            print(f"         Use --output <path.jpg> to view the annotated image.")

    fps_tracker = FPSCounter()
    fps_tracker._num_frames = 1
    fps_tracker._start = start
    fps_tracker._end = start + elapsed
    return fps_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using MobileNet SSD"
    )
    parser.add_argument(
        "-i", "--input", default="0",
        help="Path to input (.gif, .mp4, .jpg, .png) or camera index (default: 0 for webcam)"
    )
    parser.add_argument(
        "-m", "--model",
        default=os.path.join("models", "ssd_mobilenet_v1_10.onnx"),
        help="Path to model file (.onnx or .caffemodel). Default is models/ssd_mobilenet_v1_10.onnx"
    )
    parser.add_argument(
        "-p", "--prototxt",
        default=os.path.join("models", "MobileNetSSD_deploy.prototxt.txt"),
        help="Path to Caffe prototxt file (only needed if using .caffemodel)"
    )
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.5,
        help="Minimum confidence threshold (0.0 to 1.0, default: 0.5)"
    )
    parser.add_argument(
        "-w", "--width", type=int, default=600,
        help="Resize frame width (default: 600, 0 to keep original size)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Optional output file path (.gif, .mp4, .jpg, .png)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless without attempting to open a GUI window"
    )

    args = parser.parse_args()

    # Initialize Detector
    detector = ObjectDetector(
        model_path=args.model,
        prototxt_path=args.prototxt,
        min_confidence=args.confidence
    )

    target_width = args.width if args.width > 0 else None
    display = not args.no_display
    input_str = args.input

    # Run detection based on input type
    lower_input = input_str.lower()
    if lower_input.endswith(".gif"):
        fps = process_gif(input_str, detector, target_width, args.output, display)
    elif lower_input.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        fps = process_image(input_str, detector, target_width, args.output, display)
    else:
        fps = process_video_or_camera(input_str, detector, target_width, args.output, display)

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("\n[INFO] Run Summary:")
    print(f"  - Elapsed time: {fps.elapsed():.2f}s")
    print(f"  - Total frames: {fps._num_frames}")
    print(f"  - Average FPS:  {fps.fps():.2f}")


if __name__ == "__main__":
    main()
