from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, ConfigDict


EventType = Literal["entry", "exit"]
TicketStatus = Literal["active", "closed", "void"]
SpotStatus = Literal["empty", "occupied", "reserved", "maintenance"]


class VehicleEventBase(BaseModel):
    plate: str = Field(..., min_length=3)
    event_type: EventType
    spot_id: Optional[str] = None
    ticket_id: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    media_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VehicleEventCreate(VehicleEventBase):
    ...


class VehicleEventOut(VehicleEventBase):
    model_config = ConfigDict(from_attributes=True)
    id: str


class VehicleStatsPoint(BaseModel):
    day: str
    entries: int
    exits: int


class TicketBase(BaseModel):
    plate: str = Field(..., min_length=3)
    spot_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    note: Optional[str] = None
    expected_hours: Optional[float] = None


class TicketCreate(TicketBase):
    ...


class TicketUpdate(BaseModel):
    spot_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    note: Optional[str] = None
    status: Optional[TicketStatus] = None
    expected_hours: Optional[float] = None


class TicketClosePayload(BaseModel):
    payment_method: Optional[str] = None
    payment_reference: Optional[str] = None
    exit_time: Optional[datetime] = None


class TicketOut(TicketBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    ticket_id: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    status: TicketStatus = "active"
    fee: Optional[float] = None
    payment_method: Optional[str] = None
    payment_reference: Optional[str] = None


class SpotBase(BaseModel):
    spot_id: str
    zone: Optional[str] = None
    level: Optional[str] = None
    type: Optional[str] = None


class SpotCreate(SpotBase):
    status: SpotStatus = "empty"
    ticket_id: Optional[str] = None
    last_seen_at: Optional[datetime] = None
    source: Optional[str] = None


class SpotUpdate(BaseModel):
    status: Optional[SpotStatus] = None
    ticket_id: Optional[str] = None
    zone: Optional[str] = None
    level: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    last_seen_at: Optional[datetime] = None


class SpotOut(SpotCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str


class SpotDetection(BaseModel):
    spot_id: str
    status: SpotStatus
    confidence: Optional[float] = None
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    ticket_id: Optional[str] = None


class SpotSyncPayload(BaseModel):
    source: Optional[str] = None
    detections: List[SpotDetection]


class DashboardSummary(BaseModel):
    available_spots: int
    total_spots: int
    active_tickets: int
    last_entry_plate: Optional[str] = None
    last_exit_plate: Optional[str] = None
