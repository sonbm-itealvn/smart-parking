from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ReturnDocument

from ..config import SETTINGS
from ..models import SpotCreate, SpotUpdate, SpotOut, SpotSyncPayload
from ..utils.mongo import get_db


def _collection():
    return get_db()[SETTINGS.mongo_spots_collection]


def _normalize_spot_id(spot_id: str) -> str:
    return spot_id.strip().upper()


def _serialize(doc: Dict[str, Any]) -> SpotOut:
    return SpotOut(
        id=str(doc.get("_id", "")),
        spot_id=doc["spot_id"],
        status=doc.get("status", "empty"),
        ticket_id=doc.get("ticket_id"),
        zone=doc.get("zone"),
        level=doc.get("level"),
        type=doc.get("type"),
        source=doc.get("source"),
        last_seen_at=doc.get("last_seen_at"),
    )


async def create_or_update(data: SpotCreate | Dict[str, Any]) -> SpotOut:
    payload = data.dict() if isinstance(data, SpotCreate) else dict(data)
    payload["spot_id"] = _normalize_spot_id(payload["spot_id"])
    payload.setdefault("last_seen_at", datetime.utcnow())
    doc = await _collection().find_one_and_update(
        {"spot_id": payload["spot_id"]},
        {"$set": payload},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def update_spot(spot_id: str, data: SpotUpdate | Dict[str, Any]) -> Optional[SpotOut]:
    payload = data.dict(exclude_unset=True) if isinstance(data, SpotUpdate) else dict(data)
    if not payload:
        doc = await _collection().find_one({"spot_id": _normalize_spot_id(spot_id)})
        return _serialize(doc) if doc else None
    payload["last_seen_at"] = payload.get("last_seen_at", datetime.utcnow())
    doc = await _collection().find_one_and_update(
        {"spot_id": _normalize_spot_id(spot_id)},
        {"$set": payload},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc) if doc else None


async def get_spot(spot_id: str) -> Optional[SpotOut]:
    doc = await _collection().find_one({"spot_id": _normalize_spot_id(spot_id)})
    return _serialize(doc) if doc else None


async def list_spots(
    *,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[SpotOut]:
    query: Dict[str, Any] = {}
    if status:
        query["status"] = status
    cursor = (
        _collection()
        .find(query)
        .sort("spot_id", 1)
        .skip(max(skip, 0))
        .limit(min(max(limit, 1), 200))
    )
    docs = await cursor.to_list(length=None)
    return [_serialize(doc) for doc in docs]


async def available_summary() -> Dict[str, Any]:
    pipeline = [
        {
            "$group": {
                "_id": "$status",
                "count": {"$sum": 1},
            }
        }
    ]
    rows = await _collection().aggregate(pipeline).to_list(length=None)
    counts = {row["_id"]: row["count"] for row in rows}
    total = sum(counts.values())
    available = counts.get("empty", 0)
    return {"total": total, "available": available, "counts": counts}


async def reserve_spot(spot_id: str, ticket_id: str) -> Optional[SpotOut]:
    doc = await _collection().find_one_and_update(
        {"spot_id": _normalize_spot_id(spot_id)},
        {
            "$set": {
                "status": "occupied",
                "ticket_id": ticket_id,
                "last_seen_at": datetime.utcnow(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc) if doc else None


async def release_spot(spot_id: str) -> Optional[SpotOut]:
    doc = await _collection().find_one_and_update(
        {"spot_id": _normalize_spot_id(spot_id)},
        {
            "$set": {
                "status": "empty",
                "ticket_id": None,
                "last_seen_at": datetime.utcnow(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc) if doc else None


async def sync_detections(payload: SpotSyncPayload) -> List[SpotOut]:
    updates: List[SpotOut] = []
    for detection in payload.detections:
        updated = await _collection().find_one_and_update(
            {"spot_id": _normalize_spot_id(detection.spot_id)},
            {
                "$set": {
                    "spot_id": _normalize_spot_id(detection.spot_id),
                    "status": detection.status,
                    "ticket_id": detection.ticket_id,
                    "source": payload.source,
                    "last_seen_at": detection.last_seen_at,
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        updates.append(_serialize(updated))
    return updates
