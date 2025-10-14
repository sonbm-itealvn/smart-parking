from fastapi import FastAPI
from .routers.parking import router as parking_router
from .routers.license import router as license_router


def create_app() -> FastAPI:
    app = FastAPI(title="Smart Parking Backend", version="1.0.0")
    app.include_router(parking_router, prefix="/parking-space", tags=["parking-space"])
    app.include_router(license_router, prefix="/license-plate", tags=["license-plate"])
    
    @app.get("/")
    def root():
        return {"service": "Smart Parking Backend", "status": "ok"}

    return app


