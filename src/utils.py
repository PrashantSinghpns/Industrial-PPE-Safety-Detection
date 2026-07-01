"""Common utility functions for PPE detection."""

from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(weights: str, device: str = "cpu"):
    model = YOLO(weights)
    model.to(device)
    return model


def parse_detections(results):
    boxes, confs, classes = [], [], []
    for r in results:
        for b in r.boxes:
            boxes.append(b.xyxy.cpu().numpy()[0])
            confs.append(float(b.conf.cpu().numpy()[0]))
            classes.append(int(b.cls.cpu().numpy()[0]))
    if not boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    return np.array(boxes), np.array(confs), np.array(classes, dtype=int)


def draw_boxes(frame, boxes, confs, class_ids, class_names):
    for box, conf, cls in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[cls]} {conf:.2f}"
        color = (0, 255, 0) if "no" not in class_names[cls] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def detect_violations(boxes, confs, class_ids, class_names, iou_threshold=0.05):
    violations = []
    idx = {name: i for i, name in enumerate(class_names)}

    for box, conf, cls in zip(boxes, confs, class_ids):
        if class_names[cls] in ["no_helmet", "no_vest"]:
            violations.append((class_names[cls], box, conf))

    if "person" not in idx:
        return violations

    person_boxes = boxes[class_ids == idx["person"]]
    helmet_boxes = boxes[class_ids == idx.get("helmet", -1)] if "helmet" in idx else np.empty((0, 4))
    vest_boxes = boxes[class_ids == idx.get("safety_vest", -1)] if "safety_vest" in idx else np.empty((0, 4))

    for pbox in person_boxes:
        has_helmet = any(iou(pbox, hbox) > iou_threshold for hbox in helmet_boxes)
        has_vest = any(iou(pbox, vbox) > iou_threshold for vbox in vest_boxes)
        if not has_helmet:
            violations.append(("no_helmet", pbox, 0.0))
        if not has_vest:
            violations.append(("no_vest", pbox, 0.0))
    return violations
