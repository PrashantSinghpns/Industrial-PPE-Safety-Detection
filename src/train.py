"""Train YOLO model on a PPE dataset."""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPE YOLO model")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = YOLO(cfg.get("model", "yolov8s.pt"))
    model.train(
        data=cfg["dataset_yaml"],
        epochs=args.epochs,
        batch=args.batch,
        imgsz=cfg.get("img_size", 640),
        device=args.device,
        name="ppe_train",
    )


if __name__ == "__main__":
    main()
