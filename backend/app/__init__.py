from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.parking import router as parking_router
from .routers.license import router as license_router
from .routers.history import router as history_router
from .routers.tickets import router as tickets_router
from .routers.spots import router as spots_router
from .routers.dashboard import router as dashboard_router
from .utils import mongo


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await mongo.connect()
    await mongo.ensure_indexes()
    yield
    await mongo.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Smart Parking Backend", version="1.1.0", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(parking_router, prefix="/parking-space", tags=["parking-space"])
    app.include_router(license_router, prefix="/license-plate", tags=["license-plate"])

    api_prefix = "/api/v1"
    app.include_router(history_router, prefix=api_prefix)
    app.include_router(tickets_router, prefix=api_prefix)
    app.include_router(spots_router, prefix=api_prefix)
    app.include_router(dashboard_router, prefix=api_prefix)

    @app.get("/")
    def root():
        return {"service": "Smart Parking Backend", "status": "ok"}

    return app


