from __future__ import annotations

from fastapi import APIRouter

from ..models import DashboardSummary
from ..services import spot_service, ticket_service, history_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/")
async def get_dashboard():
    spots = await spot_service.available_summary()
    active_tickets = await ticket_service.count_active_tickets()
    last_entry = await history_service.latest_event("entry")
    last_exit = await history_service.latest_event("exit")
    summary = DashboardSummary(
        available_spots=spots["available"],
        total_spots=spots["total"],
        active_tickets=active_tickets,
        last_entry_plate=last_entry.plate if last_entry else None,
        last_exit_plate=last_exit.plate if last_exit else None,
    )
    data = summary.model_dump()
    data["occupancy"] = spots["counts"]
    return data
