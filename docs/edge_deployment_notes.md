# Edge Deployment Notes

## Raspberry Pi

Use a small model such as YOLOv8n. Export to ONNX and run with ONNX Runtime. Reduce image size to improve FPS.

## NVIDIA Jetson

Export model to ONNX and convert to TensorRT engine. Use FP16 or INT8 for better speed.

## Optimization Ideas

- Lower input resolution.
- Use smaller YOLO model.
- Use ONNX Runtime or TensorRT.
- Apply quantization.
- Use frame skipping if needed.
