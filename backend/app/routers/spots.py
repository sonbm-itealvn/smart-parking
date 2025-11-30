from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..models import SpotCreate, SpotUpdate, SpotSyncPayload
from ..services import spot_service

router = APIRouter(prefix="/spots", tags=["spots"])


@router.get("/")
async def list_spots(status: str | None = Query(None, pattern="^(empty|occupied|reserved|maintenance)$"), skip: int = 0, limit: int = 100):
    spots = await spot_service.list_spots(status=status, skip=skip, limit=limit)
    return {"spots": [spot.model_dump() for spot in spots]}


@router.post("/", status_code=201)
async def create_spot(payload: SpotCreate):
    spot = await spot_service.create_or_update(payload)
    return spot.model_dump()


@router.patch("/{spot_id}")
async def update_spot(spot_id: str, payload: SpotUpdate):
    spot = await spot_service.update_spot(spot_id, payload)
    if not spot:
        raise HTTPException(status_code=404, detail="Spot not found")
    return spot.model_dump()


@router.get("/available")
async def available_summary():
    summary = await spot_service.available_summary()
    return summary


@router.post("/sync")
async def sync_spots(payload: SpotSyncPayload):
    spots = await spot_service.sync_detections(payload)
    return {"updated": [spot.model_dump() for spot in spots]}


@router.post("/{spot_id}/release")
async def release_spot(spot_id: str):
    spot = await spot_service.release_spot(spot_id)
    if not spot:
        raise HTTPException(status_code=404, detail="Spot not found")
    return spot.model_dump()
