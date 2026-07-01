"""CSV logging utilities for PPE violation events."""

import csv
from datetime import datetime
from pathlib import Path


HEADER = ["timestamp", "violation_type", "confidence", "source"]


def init_log(csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)


def log_violation(csv_path: str | Path, violation_type: str, confidence: float, source: str) -> None:
    init_log(csv_path)
    with Path(csv_path).open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            violation_type,
            f"{confidence:.2f}",
            source,
        ])
