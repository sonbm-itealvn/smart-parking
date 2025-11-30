from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..models import TicketCreate, TicketUpdate, TicketClosePayload
from ..services import ticket_service

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.get("/")
async def list_tickets(
    status: str | None = Query(None, pattern="^(active|closed|void)$"),
    plate: str | None = None,
    skip: int = 0,
    limit: int = 50,
):
    tickets = await ticket_service.list_tickets(status=status, plate=plate, skip=skip, limit=limit)
    return {"tickets": [ticket.model_dump() for ticket in tickets]}


@router.post("/", status_code=201)
async def create_ticket(payload: TicketCreate):
    try:
        ticket = await ticket_service.create_ticket(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ticket.model_dump()


@router.get("/{ticket_id}")
async def get_ticket(ticket_id: str):
    ticket = await ticket_service.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket.model_dump()


@router.patch("/{ticket_id}")
async def update_ticket(ticket_id: str, payload: TicketUpdate):
    ticket = await ticket_service.update_ticket(ticket_id, payload)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket.model_dump()


@router.post("/{ticket_id}/close")
async def close_ticket(ticket_id: str, payload: TicketClosePayload):
    ticket = await ticket_service.close_ticket(ticket_id, payload)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket.model_dump()


@router.get("/active/{plate}")
async def active_ticket_for_plate(plate: str):
    ticket = await ticket_service.find_active_ticket_by_plate(plate)
    if not ticket:
        raise HTTPException(status_code=404, detail="No active ticket for plate")
    return ticket.model_dump()
