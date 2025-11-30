from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId

from ..config import SETTINGS
from ..models import VehicleEventCreate, VehicleEventOut, VehicleStatsPoint
from ..utils.mongo import get_db
from .license_logs import normalize_plate


def _collection():
    db = get_db()
    return db[SETTINGS.mongo_vehicle_events_collection]


def _serialize(doc: Dict[str, Any]) -> VehicleEventOut:
    return VehicleEventOut(
        id=str(doc.get("_id", "")),
        plate=doc["plate"],
        event_type=doc["event_type"],
        spot_id=doc.get("spot_id"),
        ticket_id=doc.get("ticket_id"),
        source=doc.get("source"),
        confidence=doc.get("confidence"),
        media_url=doc.get("media_url"),
        timestamp=doc["timestamp"],
    )


async def record_event(payload: VehicleEventCreate | Dict[str, Any]) -> VehicleEventOut:
    data = payload.dict() if isinstance(payload, VehicleEventCreate) else dict(payload)
    data["plate"] = normalize_plate(data["plate"])
    data.setdefault("timestamp", datetime.utcnow())
    result = await _collection().insert_one(data)
    data["_id"] = result.inserted_id
    return _serialize(data)


async def list_events(
    *,
    plate: Optional[str] = None,
    event_type: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[VehicleEventOut]:
    query: Dict[str, Any] = {}
    if plate:
        query["plate"] = normalize_plate(plate)
    if event_type:
        query["event_type"] = event_type
    if start or end:
        query["timestamp"] = {}
        if start:
            query["timestamp"]["$gte"] = start
        if end:
            query["timestamp"]["$lte"] = end
    cursor = (
        _collection()
        .find(query)
        .sort("timestamp", -1)
        .skip(max(skip, 0))
        .limit(min(max(limit, 1), 200))
    )
    docs = await cursor.to_list(length=None)
    return [_serialize(doc) for doc in docs]


async def latest_by_plate(plate: str) -> Optional[VehicleEventOut]:
    doc = await (
        _collection()
        .find({"plate": normalize_plate(plate)})
        .sort("timestamp", -1)
        .limit(1)
        .to_list(length=1)
    )
    if not doc:
        return None
    return _serialize(doc[0])


async def latest_event(event_type: Optional[str] = None) -> Optional[VehicleEventOut]:
    query: Dict[str, Any] = {}
    if event_type:
        query["event_type"] = event_type
    docs = await _collection().find(query).sort("timestamp", -1).limit(1).to_list(length=1)
    if not docs:
        return None
    return _serialize(docs[0])


async def stats_by_day(days: int = 7) -> List[VehicleStatsPoint]:
    pipeline = [
        {
            "$project": {
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "event_type": 1,
            }
        },
        {
            "$group": {
                "_id": {"day": "$day", "event_type": "$event_type"},
                "count": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.day",
                "counts": {
                    "$push": {"event_type": "$_id.event_type", "count": "$count"},
                },
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": days},
    ]
    raw = await _collection().aggregate(pipeline).to_list(length=days)
    stats: List[VehicleStatsPoint] = []
    for doc in raw:
        counts = {"entry": 0, "exit": 0}
        for bucket in doc.get("counts", []):
            counts[bucket["event_type"]] = bucket["count"]
        stats.append(
            VehicleStatsPoint(
                day=str(doc["_id"]),
                entries=counts["entry"],
                exits=counts["exit"],
            )
        )
    stats.sort(key=lambda item: item.day)
    return stats


async def delete_event(event_id: str) -> bool:
    try:
        obj_id = ObjectId(event_id)
    except InvalidId:
        return False
    result = await _collection().delete_one({"_id": obj_id})
    return result.deleted_count > 0
