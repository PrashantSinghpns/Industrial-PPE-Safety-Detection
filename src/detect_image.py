"""Run PPE detection on one image."""

import argparse
from pathlib import Path
import cv2

from logger import init_log
from utils import load_config, load_model, parse_detections, draw_boxes
from violation_alert import handle_violations


def main():
    parser = argparse.ArgumentParser(description="PPE detection on image")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    output_dir = Path(cfg.get("output_dir", "outputs"))
    log_path = output_dir / "violation_logs.csv"
    screenshots_dir = output_dir / "screenshots"
    init_log(log_path)

    image = cv2.imread(str(args.source))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    model = load_model(args.weights, cfg.get("device", "cpu"))
    results = model.predict(image, conf=cfg.get("conf_threshold", 0.25), iou=cfg.get("iou_threshold", 0.45), verbose=False)
    boxes, confs, class_ids = parse_detections(results)

    handle_violations(image, boxes, confs, class_ids, classes, log_path, screenshots_dir, args.source.stem)
    annotated = draw_boxes(image.copy(), boxes, confs, class_ids, classes)

    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{args.source.stem}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
