# Industrial PPE & Safety Violation Detection System

An industry-focused computer vision project for detecting PPE compliance in industrial environments using Python, OpenCV and YOLO.

The system detects workers, helmets and safety vests from images, videos or webcam streams. It generates violation alerts, saves screenshots and logs safety incidents into a CSV file.

## Industrial Use Case

Factories, mining sites, steel plants, airports and construction zones need continuous safety monitoring. Manual supervision is not always reliable. This project helps detect missing PPE such as helmets or safety vests and creates evidence logs for compliance review.

## Features

- Detect people, helmets, no-helmet, safety vest and no-vest classes.
- Run inference on images, videos and webcam streams.
- Draw bounding boxes and class labels.
- Generate PPE violation alerts.
- Save violation screenshots.
- Log violations into CSV with timestamp, type, confidence and source.
- Show basic FPS progress during video/webcam inference.
- Support YOLO training and ONNX export.
- Include edge deployment notes for Raspberry Pi and NVIDIA Jetson.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas
- PyYAML
- PyTorch backend through Ultralytics
- CSV logging
- Optional ONNX export

## Folder Structure

```text
Industrial-PPE-Safety-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   ├── README.md
│   └── dataset_instructions.md
├── src/
│   ├── train.py
│   ├── detect_image.py
│   ├── file_inference.py
│   ├── realtime_webcam.py
│   ├── violation_alert.py
│   ├── logger.py
│   ├── utils.py
│   └── model_conversion.py
├── outputs/
│   ├── screenshots/
│   ├── videos/
│   └── violation_logs.csv
├── docs/
│   ├── project_explanation.md
│   ├── interview_questions.md
│   ├── metrics_explanation.md
│   └── edge_deployment_notes.md
└── assets/
    └── sample_images/
```

## Dataset Setup

Use a public PPE dataset in YOLO format. Suitable sources include Roboflow PPE datasets or the Ultralytics Construction PPE dataset.

Expected classes:

- `person`
- `helmet`
- `no_helmet`
- `safety_vest`
- `no_vest`

Example dataset YAML:

```yaml
train: data/ppe_dataset/images/train
val: data/ppe_dataset/images/val
nc: 5
names: [person, helmet, no_helmet, safety_vest, no_vest]
```

Update `config.yaml` after placing your dataset.

## Installation

```bash
git clone https://github.com/PrashantSinghpns/Industrial-PPE-Safety-Detection.git
cd Industrial-PPE-Safety-Detection
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --config config.yaml --epochs 50 --batch 16
```

After training, use the generated weights from the `runs/train/` directory.

## Image Detection

```bash
python src/detect_image.py --config config.yaml --weights runs/train/ppe_train/weights/best.pt --source assets/sample_images/test.jpg --save
```

## Video Detection

```bash
python src/detect_video.py --config config.yaml --weights runs/train/ppe_train/weights/best.pt --source input_video.mp4 --save
```

## Webcam Detection

```bash
python src/realtime_webcam.py --config config.yaml --weights runs/train/ppe_train/weights/best.pt --camera 0 --save
```

## ONNX Export

```bash
python src/export_onnx.py --weights runs/train/ppe_train/weights/best.pt --output models/ppe_model.onnx --simplify
```

Use ONNX for edge deployment with ONNX Runtime, TensorRT or other optimised inference engines.

## Output Examples

The project creates:

- Annotated images and videos.
- Violation screenshots in `outputs/screenshots/`.
- CSV logs in `outputs/violation_logs.csv`.

CSV columns:

```text
timestamp, violation_type, confidence, source
```

## Metrics Explanation

Important evaluation metrics:

- **Precision**: how many predicted detections are correct.
- **Recall**: how many actual objects are detected.
- **mAP**: overall detection quality across classes.
- **Confusion matrix**: shows class-wise mistakes.

See `docs/metrics_explanation.md` for details.

## Edge Deployment

For Raspberry Pi or Jetson deployment:

- Use a small YOLO model such as YOLOv8n.
- Reduce image size to improve FPS.
- Export the model to ONNX.
- Use ONNX Runtime on Raspberry Pi.
- Use TensorRT on NVIDIA Jetson.
- Consider FP16 or INT8 quantization.

See `docs/edge_deployment_notes.md` for details.

## Resume Bullet Points

- Developed an industrial PPE safety violation detection system using Python, OpenCV and YOLO for real-time worker safety monitoring.
- Implemented helmet and safety vest detection with violation alert generation, screenshot capture and CSV-based compliance logging.
- Optimised inference pipeline with confidence thresholding, FPS monitoring and ONNX export support for edge deployment on Raspberry Pi/Jetson-class devices.

## Explanation

This project demonstrates computer vision, object detection, dataset handling, model inference, edge deployment thinking and clean Python engineering. It is aligned with industrial AI-IoT roles where computer vision is used for safety, compliance and productivity monitoring.

## Future Improvements

- Add object tracking using ByteTrack or DeepSORT.
- Add restricted zone intrusion detection.
- Add OCR for worker ID or safety sign recognition.
- Add dashboard for violation analytics.
- Add MQTT alerts for IoT integration.
- Deploy on Raspberry Pi or NVIDIA Jetson.
