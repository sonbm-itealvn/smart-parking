from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional
import os


@dataclass(frozen=True)
class Settings:
    """Central configuration for models, logging, and heuristics."""

    repo_root: Path = Path(__file__).resolve().parents[2]
    parking_model_path: Path = repo_root / "models" / "parking" / "best.pt"
    license_model_path: Path = repo_root / "models" / "license_plate" / "best.pt"
    license_log_path: Path = repo_root / "tests" / "license_plate" / "log.csv"
    license_confidence: float = float(os.getenv("LICENSE_CONFIDENCE", "0.6"))
    parking_confidence: float = float(os.getenv("PARKING_CONFIDENCE", "0.6"))
    parking_entry_point: Tuple[int, int] = (110, 5)
    parking_max_width: int = 960
    parking_landmarks: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: {
            0: (110, 5),
            1: (100, 100),
            2: (80, 165),
            3: (60, 300),
        }
    )
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "smart_parking")
    mongo_vehicle_events_collection: str = os.getenv("MONGO_VEHICLE_EVENTS_COLLECTION", "vehicle_events")
    mongo_tickets_collection: str = os.getenv("MONGO_TICKETS_COLLECTION", "tickets")
    mongo_spots_collection: str = os.getenv("MONGO_SPOTS_COLLECTION", "parking_spots")
    mongo_active_vehicles_collection: str = os.getenv("MONGO_ACTIVE_VEHICLES_COLLECTION", "active_vehicles")
    ticket_rate_hour: float = float(os.getenv("TICKET_RATE_HOUR", "10000"))
    ticket_first_hour_rate: Optional[float] = (
        float(os.getenv("TICKET_FIRST_HOUR_RATE")) if os.getenv("TICKET_FIRST_HOUR_RATE") else None
    )
    ticket_max_daily_rate: Optional[float] = (
        float(os.getenv("TICKET_MAX_DAILY_RATE")) if os.getenv("TICKET_MAX_DAILY_RATE") else None
    )


SETTINGS = Settings()
