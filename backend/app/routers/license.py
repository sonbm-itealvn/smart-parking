from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
import io
from datetime import datetime

from ..services.license_plate import detect_license_plates
from ..services.license_logs import read_logs
from ..models import VehicleEventCreate
from ..services import history_service, presence_service
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes

router = APIRouter()


@router.post("/detect")
async def detect(
    image: UploadFile = File(...),
    event_type: str | None = Query(None, pattern="^(entry|exit)$"),
    spot_id: str | None = None,
    source: str | None = None,
):
    try:
        data = await image.read()
        frame_bgr = bytes_to_cv2_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = detect_license_plates(frame_bgr, log_results=True)
    timestamp = datetime.utcnow()
    for plate in result.texts:
        effective_event = event_type
        if effective_event is None:
            is_inside = await presence_service.is_present(plate)
            effective_event = "exit" if is_inside else "entry"
        event = await history_service.record_event(
            VehicleEventCreate(
                plate=plate,
                event_type=effective_event,
                spot_id=spot_id,
                source=source or "license-camera",
                timestamp=timestamp,
            )
        )
        if effective_event == "entry":
            await presence_service.mark_entry(
                event.plate,
                event_id=event.id,
                spot_id=spot_id,
                source=source or "license-camera",
                timestamp=event.timestamp,
            )
        else:
            await presence_service.mark_exit(event.plate, timestamp=event.timestamp)

    png_bytes = cv2_image_to_png_bytes(result.annotated_bgr)
    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"X-Recognized-Texts": ";".join(result.texts)},
    )


@router.get("/logs")
def logs():
    return {"logs": read_logs()}


