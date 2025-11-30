from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..models import VehicleEventCreate
from ..services import history_service, presence_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/")
async def get_history(
    plate: Optional[str] = None,
    event_type: Optional[str] = Query(None, pattern="^(entry|exit)$"),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 50,
):
    events = await history_service.list_events(
        plate=plate,
        event_type=event_type,
        start=start,
        end=end,
        skip=skip,
        limit=limit,
    )
    return {"events": [event.model_dump() for event in events]}


@router.post("/", status_code=201)
async def create_history_entry(payload: VehicleEventCreate):
    event = await history_service.record_event(payload)
    if event.event_type == "entry":
        await presence_service.mark_entry(
            event.plate,
            event_id=event.id,
            spot_id=event.spot_id,
            ticket_id=event.ticket_id,
            source=event.source,
            timestamp=event.timestamp,
        )
    else:
        await presence_service.mark_exit(event.plate, timestamp=event.timestamp)
    return event.model_dump()


@router.get("/{plate}/latest")
async def latest_for_plate(plate: str):
    event = await history_service.latest_by_plate(plate)
    if not event:
        raise HTTPException(status_code=404, detail="No history for plate")
    return event.model_dump()


@router.get("/stats/daily")
async def stats_daily(days: int = Query(7, ge=1, le=30)):
    stats = await history_service.stats_by_day(days)
    return {"stats": [point.model_dump() for point in stats]}


@router.delete("/{event_id}")
async def delete_event(event_id: str):
    deleted = await history_service.delete_event(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"deleted": True}
