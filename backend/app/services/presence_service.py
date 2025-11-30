from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import SETTINGS
from ..utils.mongo import get_db
from .license_logs import normalize_plate


def _collection():
    return get_db()[SETTINGS.mongo_active_vehicles_collection]


async def is_present(plate: str) -> bool:
    normalized = normalize_plate(plate)
    doc = await _collection().find_one({"plate": normalized})
    return doc is not None


async def mark_entry(
    plate: str,
    *,
    event_id: Optional[str] = None,
    spot_id: Optional[str] = None,
    ticket_id: Optional[str] = None,
    source: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    normalized = normalize_plate(plate)
    payload = {
        "plate": normalized,
        "event_id": event_id,
        "spot_id": spot_id,
        "ticket_id": ticket_id,
        "source": source,
        "last_seen_at": timestamp or datetime.utcnow(),
    }
    await _collection().update_one({"plate": normalized}, {"$set": payload}, upsert=True)


async def mark_exit(
    plate: str,
    *,
    timestamp: Optional[datetime] = None,
) -> None:
    normalized = normalize_plate(plate)
    await _collection().delete_one({"plate": normalized})


async def active_count() -> int:
    return await _collection().count_documents({})


async def list_active():
    cursor = _collection().find().sort("plate", 1)
    return await cursor.to_list(length=None)
