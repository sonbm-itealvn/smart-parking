from __future__ import annotations

from datetime import datetime
import math
import uuid
from typing import Any, Dict, List, Optional

from pymongo import ReturnDocument

from ..config import SETTINGS
from ..models import (
    TicketCreate,
    TicketUpdate,
    TicketOut,
    TicketClosePayload,
    VehicleEventCreate,
)
from ..utils.mongo import get_db
from .license_logs import normalize_plate
from . import history_service, spot_service, presence_service


def _collection():
    return get_db()[SETTINGS.mongo_tickets_collection]


def _serialize(doc: Dict[str, Any]) -> TicketOut:
    return TicketOut(
        id=str(doc.get("_id", "")),
        ticket_id=doc["ticket_id"],
        plate=doc["plate"],
        spot_id=doc.get("spot_id"),
        vehicle_type=doc.get("vehicle_type"),
        note=doc.get("note"),
        expected_hours=doc.get("expected_hours"),
        entry_time=doc["entry_time"],
        exit_time=doc.get("exit_time"),
        status=doc.get("status", "active"),
        fee=doc.get("fee"),
        payment_method=doc.get("payment_method"),
        payment_reference=doc.get("payment_reference"),
    )


def _generate_ticket_id() -> str:
    return uuid.uuid4().hex[:10].upper()


def _compute_fee(entry_time: datetime, exit_time: datetime) -> float:
    duration_hours = max((exit_time - entry_time).total_seconds() / 3600.0, 0.0)
    if duration_hours <= 0:
        return 0.0
    first_rate = SETTINGS.ticket_first_hour_rate or SETTINGS.ticket_rate_hour
    total = first_rate
    if duration_hours > 1:
        total += math.ceil(duration_hours - 1) * SETTINGS.ticket_rate_hour
    if SETTINGS.ticket_max_daily_rate:
        total = min(total, SETTINGS.ticket_max_daily_rate)
    return float(total)


async def _ensure_spot_assignment(spot_id: Optional[str], ticket_id: str) -> None:
    if not spot_id:
        return
    existing = await spot_service.get_spot(spot_id)
    if existing is None:
        await spot_service.create_or_update(
            {
                "spot_id": spot_id,
                "status": "occupied",
                "ticket_id": ticket_id,
            }
        )
    else:
        await spot_service.reserve_spot(spot_id, ticket_id)


async def create_ticket(payload: TicketCreate) -> TicketOut:
    now = datetime.utcnow()
    data = payload.dict()
    data["plate"] = normalize_plate(data["plate"])
    existing = await _collection().find_one({"plate": data["plate"], "status": "active"})
    if existing:
        raise ValueError("Vehicle already has an active ticket")
    data["ticket_id"] = _generate_ticket_id()
    data["entry_time"] = now
    data["status"] = "active"
    data["fee"] = None
    result = await _collection().insert_one(data)
    data["_id"] = result.inserted_id

    await _ensure_spot_assignment(data.get("spot_id"), data["ticket_id"])
    event = await history_service.record_event(
        VehicleEventCreate(
            plate=data["plate"],
            ticket_id=data["ticket_id"],
            spot_id=data.get("spot_id"),
            event_type="entry",
            source="ticket",
            timestamp=now,
        )
    )
    await presence_service.mark_entry(
        event.plate,
        event_id=event.id,
        spot_id=event.spot_id,
        ticket_id=event.ticket_id,
        source=event.source,
        timestamp=event.timestamp,
    )
    return _serialize(data)


async def list_tickets(
    *,
    status: Optional[str] = None,
    plate: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> List[TicketOut]:
    query: Dict[str, Any] = {}
    if status:
        query["status"] = status
    if plate:
        query["plate"] = normalize_plate(plate)
    cursor = (
        _collection()
        .find(query)
        .sort("entry_time", -1)
        .skip(max(skip, 0))
        .limit(min(max(limit, 1), 200))
    )
    docs = await cursor.to_list(length=None)
    return [_serialize(doc) for doc in docs]


async def get_ticket(ticket_id: str) -> Optional[TicketOut]:
    doc = await _collection().find_one({"ticket_id": ticket_id})
    return _serialize(doc) if doc else None


async def update_ticket(ticket_id: str, payload: TicketUpdate) -> Optional[TicketOut]:
    data = payload.dict(exclude_unset=True)
    if not data:
        return await get_ticket(ticket_id)
    doc = await _collection().find_one_and_update(
        {"ticket_id": ticket_id},
        {"$set": data},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc) if doc else None


async def close_ticket(ticket_id: str, payload: TicketClosePayload) -> Optional[TicketOut]:
    ticket = await _collection().find_one({"ticket_id": ticket_id})
    if ticket is None:
        return None
    if ticket.get("status") == "closed":
        return _serialize(ticket)

    exit_time = payload.exit_time or datetime.utcnow()
    fee = _compute_fee(ticket["entry_time"], exit_time)
    updates = {
        "status": "closed",
        "exit_time": exit_time,
        "fee": fee,
        "payment_method": payload.payment_method,
        "payment_reference": payload.payment_reference,
    }
    doc = await _collection().find_one_and_update(
        {"ticket_id": ticket_id},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if ticket.get("spot_id"):
        await spot_service.release_spot(ticket["spot_id"])
    event = await history_service.record_event(
        VehicleEventCreate(
            plate=ticket["plate"],
            ticket_id=ticket_id,
            spot_id=ticket.get("spot_id"),
            event_type="exit",
            source="ticket",
            timestamp=exit_time,
        )
    )
    await presence_service.mark_exit(event.plate, timestamp=event.timestamp)
    return _serialize(doc) if doc else None


async def find_active_ticket_by_plate(plate: str) -> Optional[TicketOut]:
    doc = await _collection().find_one({"plate": normalize_plate(plate), "status": "active"})
    return _serialize(doc) if doc else None


async def count_active_tickets() -> int:
    return await _collection().count_documents({"status": "active"})
