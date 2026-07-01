"""Violation alert handling for PPE detection."""

from pathlib import Path
import cv2

from logger import log_violation
from utils import detect_violations


def handle_violations(frame, boxes, confs, class_ids, class_names, log_path, screenshots_dir, source_id):
    screenshots_dir = Path(screenshots_dir)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    violations = detect_violations(boxes, confs, class_ids, class_names)
    for count, (violation_type, box, conf) in enumerate(violations, start=1):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
        if crop.size > 0:
            shot_path = screenshots_dir / f"{violation_type}_{source_id}_{count}.jpg"
            cv2.imwrite(str(shot_path), crop)
        log_violation(log_path, violation_type, float(conf), source_id)
