"""Run PPE detection on a video file.

Example:
    python src/file_inference.py --weights runs/train/ppe_train/weights/best.pt --source input.mp4 --save
"""

import argparse
from pathlib import Path
import time
import cv2

from logger import init_log
from utils import load_config, load_model, parse_detections, draw_boxes
from violation_alert import handle_violations


def main():
    parser = argparse.ArgumentParser(description="PPE detection on a video file")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--source", type=Path, required=True, help="Path to input video")
    parser.add_argument("--save", action="store_true", help="Save processed output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    output_dir = Path(cfg.get("output_dir", "outputs"))
    log_path = output_dir / "violation_logs.csv"
    screenshots_dir = output_dir / "screenshots"
    init_log(log_path)

    model = load_model(args.weights, cfg.get("device", "cpu"))

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {args.source}")

    writer = None
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        save_path = output_dir / f"{args.source.stem}_detected.mp4"
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(
            frame,
            conf=cfg.get("conf_threshold", 0.25),
            iou=cfg.get("iou_threshold", 0.45),
            verbose=False,
        )

        boxes, confs, class_ids = parse_detections(results)
        handle_violations(
            frame,
            boxes,
            confs,
            class_ids,
            class_names,
            log_path,
            screenshots_dir,
            f"{args.source.stem}_frame_{frame_count}",
        )

        annotated = draw_boxes(frame.copy(), boxes, confs, class_ids, class_names)
        fps_now = frame_count / max(time.time() - start_time, 1e-6)
        cv2.putText(
            annotated,
            f"FPS: {fps_now:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        if writer is not None:
            writer.write(annotated)

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output video: {save_path}")


if __name__ == "__main__":
    main()
