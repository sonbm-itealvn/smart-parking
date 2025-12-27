from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io
import base64
from datetime import datetime

from ..services.license_plate import detect_license_plates
from ..services.license_logs import read_logs
from ..models import VehicleEventCreate
from ..services import history_service, presence_service
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes, url_to_cv2_image

router = APIRouter()


@router.post("/detect")
async def detect(
    image: UploadFile | None = File(None),
    image_url: str | None = Query(None, description="URL of the image to process"),
    event_type: str | None = Query(None, pattern="^(entry|exit)$"),
    spot_id: str | None = None,
    source: str | None = None,
    format: str = Query("json", pattern="^(json|image)$"),
):
    """
    Detect license plates in an uploaded image or from an image URL.
    
    Either `image` (file upload) or `image_url` must be provided.
    
    - **image**: Uploaded image file (optional if image_url is provided)
    - **image_url**: URL of the image to fetch and process (optional if image is provided)
    - **format**: Response format - "json" (default) returns JSON with text and image, "image" returns PNG image only
    - **event_type**: Optional event type ("entry" or "exit"). If not provided, auto-detects based on presence
    - **spot_id**: Optional parking spot ID
    - **source**: Optional source identifier
    """
    # Validate that at least one input method is provided
    if image is None and image_url is None:
        raise HTTPException(
            status_code=400, 
            detail="Either 'image' (file upload) or 'image_url' parameter must be provided"
        )
    
    if image is not None and image_url is not None:
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'image' (file upload) OR 'image_url', not both"
        )
    
    try:
        if image_url:
            # Fetch image from URL
            frame_bgr = url_to_cv2_image(image_url)
        else:
            # Read uploaded file
            data = await image.read()
            frame_bgr = bytes_to_cv2_image(data)
    except Exception as e:
        error_msg = str(e)
        if image_url:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to fetch or process image from URL '{image_url}': {error_msg}"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid image: {error_msg}")

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

    # Return JSON format by default (includes text in response body)
    if format == "json":
        png_bytes = cv2_image_to_png_bytes(result.annotated_bgr)
        image_base64 = base64.b64encode(png_bytes).decode("utf-8")
        
        # Format response với thông tin chi tiết từng biển số
        plates_detail = [
            {
                "text": detail.text,
                "confidence": round(detail.confidence, 3),
                "detection_confidence": round(detail.detection_confidence, 3),
                "bbox": detail.bbox,  # [x1, y1, x2, y2]
            }
            for detail in result.details
        ]
        
        return JSONResponse(
            content={
                "success": True,
                "count": len(result.texts),
                "plates": result.texts,  # Giữ để backward compatibility
                "details": plates_detail,  # Thông tin chi tiết từng biển số
                "image": f"data:image/png;base64,{image_base64}",  # Ảnh đã detect với bounding boxes
            }
        )
    
    # Legacy image format (for backward compatibility)
    png_bytes = cv2_image_to_png_bytes(result.annotated_bgr)
    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"X-Recognized-Texts": ";".join(result.texts)},
    )


@router.get("/logs")
def logs():
    return {"logs": read_logs()}


