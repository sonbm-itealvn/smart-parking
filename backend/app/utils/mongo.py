from __future__ import annotations

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from ..config import SETTINGS

_client: Optional[AsyncIOMotorClient] = None


async def connect() -> AsyncIOMotorClient:
    """Create a singleton Mongo client for the app lifecycle."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(SETTINGS.mongo_uri)
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("Mongo client is not initialized. Call connect() first.")
    return _client


def get_db(name: str | None = None) -> AsyncIOMotorDatabase:
    return get_client()[name or SETTINGS.mongo_db]


def get_collection(name: str) -> AsyncIOMotorCollection:
    return get_db()[name]


async def ensure_indexes() -> None:
    """Create the indexes required for the core collections."""
    db = get_db()
    events = db[SETTINGS.mongo_vehicle_events_collection]
    tickets = db[SETTINGS.mongo_tickets_collection]
    spots = db[SETTINGS.mongo_spots_collection]
    active = db[SETTINGS.mongo_active_vehicles_collection]

    await events.create_index([("plate", 1), ("timestamp", -1)])
    await events.create_index("event_type")

    await tickets.create_index("ticket_id", unique=True)
    await tickets.create_index("plate")
    await tickets.create_index("status")
    await tickets.create_index([("status", 1), ("entry_time", -1)])

    await spots.create_index("spot_id", unique=True)
    await spots.create_index("status")

    await active.create_index("plate", unique=True)
