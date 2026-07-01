"""Run real-time PPE detection using a webcam.

Example:
    python src/realtime_webcam.py --weights runs/train/ppe_train/weights/best.pt --camera 0
"""

import argparse
from pathlib import Path
import time
import cv2

from logger import init_log
from utils import load_config, load_model, parse_detections, draw_boxes
from violation_alert import handle_violations


def main():
    parser = argparse.ArgumentParser(description="Realtime PPE detection using webcam")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--save", action="store_true", help="Save processed stream")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["classes"]
    output_dir = Path(cfg.get("output_dir", "outputs"))
    log_path = output_dir / "violation_logs.csv"
    screenshots_dir = output_dir / "screenshots"
    init_log(log_path)

    model = load_model(args.weights, cfg.get("device", "cpu"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    writer = None
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        save_path = output_dir / "webcam_detected.mp4"
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
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
                f"webcam_frame_{frame_count}",
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

            cv2.imshow("PPE Safety Detection", annotated)
            if writer is not None:
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
