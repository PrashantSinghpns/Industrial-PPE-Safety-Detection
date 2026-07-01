# Interview Questions

## Why YOLO?

YOLO is fast and suitable for real-time object detection.

## What is object detection?

Object detection finds object location and class using bounding boxes.

## What is mAP?

mAP measures detection quality across classes.

## What are precision and recall?

Precision measures correct predictions. Recall measures detected ground-truth objects.

## How does violation logic work?

The system checks whether a detected person has helmet and vest detections nearby. Missing PPE is logged.

## How will you improve FPS?

Use smaller models, lower image size, ONNX/TensorRT and quantization.

## How will you deploy on Raspberry Pi?

Export model to ONNX, use ONNX Runtime and reduce input size.

## How will you reduce false positives?

Improve dataset quality, tune confidence threshold and validate detections across multiple frames.

## What is ONNX?

ONNX is a portable model format for running ML models across different runtimes.

## What is quantization?

Quantization reduces model precision to improve speed and reduce memory usage.
