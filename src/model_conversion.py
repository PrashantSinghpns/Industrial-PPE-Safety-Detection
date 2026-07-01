"""Convert trained YOLO weights to ONNX.

Example:
    python src/model_conversion.py --weights runs/train/ppe_train/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export trained YOLO model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
    )
    print("ONNX export completed successfully")


if __name__ == "__main__":
    main()
