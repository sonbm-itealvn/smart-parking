from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import SETTINGS


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_plate(text: str) -> str:
    """Normalize plate format by removing spaces and dashes."""
    cleaned = "".join(part for part in text.strip().replace("-", " ").split())
    return cleaned.upper()


def append_plate(plate: str, timestamp: Optional[datetime] = None) -> None:
    plate = normalize_plate(plate)
    if not plate:
        return
    timestamp = timestamp or datetime.now()
    path = SETTINGS.license_log_path
    _ensure_parent(path)
    existing = {entry["plate"] for entry in read_logs()}
    if plate in existing:
        return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([plate, timestamp.isoformat()])


def read_logs() -> List[dict]:
    path = SETTINGS.license_log_path
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append({"plate": row[0], "timestamp": row[1] if len(row) > 1 else None})
    return rows

