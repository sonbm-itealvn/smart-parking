from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
import io
import cv2
import tempfile
from typing import Optional

from ..services.parking_service import (
    recommend_from_frame,
    recommend_from_video,
    annotate_video_file,
    detect_vehicles,
    detect_vehicles_pretrained,
)
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes


router = APIRouter()


@router.post("/detect-vehicles")
async def detect_vehicles_endpoint(
    image: UploadFile = File(..., description="Image file containing vehicles"),
    use_pretrained: bool = Query(
        False,
        description="Use YOLOv8x pretrained (COCO) for detection instead of custom parking model",
    ),
):
    """
    Detect vehicles and return their bounding boxes.
    Returns JSON with list of detections: bbox [x1, y1, x2, y2], confidence, class_id, class_name.
    """
    try:
        data = await image.read()
        frame_bgr = bytes_to_cv2_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    detections = (
        detect_vehicles_pretrained(frame_bgr)
        if use_pretrained
        else detect_vehicles(frame_bgr)
    )
    return {
        "count": len(detections),
        "detections": [
            {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
            for det in detections
        ],
        "model": "yolov8x.pt (pretrained COCO)" if use_pretrained else "custom parking model",
    }


@router.post("/recommend")
async def recommend(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    video: UploadFile = File(None),
    frame_stride: int = 5,
    max_frames: int = 300,
):
    # Determine which input is provided
    src = file or image or video
    if src is None:
        raise HTTPException(status_code=400, detail="Provide 'file' (image/video) or 'image' or 'video' in form-data")

    ctype = (src.content_type or "").lower()
    is_image = ctype.startswith("image/") or (src is image and image is not None and video is None)
    is_video = ctype.startswith("video/") or (src is video and video is not None)

    if is_image and is_video:
        raise HTTPException(status_code=400, detail="Ambiguous upload. Provide only one of image or video.")

    if is_image or (not is_video):
        # Default to image if content-type is unknown
        try:
            data = await src.read()
            frame_bgr = bytes_to_cv2_image(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        result = recommend_from_frame(frame_bgr)
        if result.routed_bgr is None:
            return JSONResponse({"message": result.message or "All parking spaces are full"})

        png_bytes = cv2_image_to_png_bytes(result.routed_bgr)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

    # video path
    try:
        data = await src.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        result = recommend_from_video(cap, frame_stride=frame_stride, max_frames=max_frames)
        cap.release()

    if result.routed_bgr is None:
        return JSONResponse({"message": result.message or "No empty parking space found in video within limits"})

    png_bytes = cv2_image_to_png_bytes(result.routed_bgr)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@router.post("/recommend-video")
async def recommend_video(video: UploadFile = File(...), frame_stride: int = 5, max_frames: int = 300):
    try:
        data = await video.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}")

    # Persist to temp file so OpenCV can read
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        result = recommend_from_video(cap, frame_stride=frame_stride, max_frames=max_frames)
        cap.release()

    if result.routed_bgr is None:
        return JSONResponse({"message": result.message or "No empty parking space found in video within limits"})

    routed_png = cv2.imencode(".png", result.routed_bgr)[1].tobytes()
    return StreamingResponse(io.BytesIO(routed_png), media_type="image/png")


@router.post("/annotate-video")
async def annotate_video(
    video: UploadFile | None = File(
        None, description="Video file to process (preferred field name: 'video')"
    ),
    file: UploadFile | None = File(
        None, description="Alternative field name for video file (e.g. some clients use 'file')"
    ),
    frame_stride: int = Query(
        1,
        ge=1,
        le=30,
        description="Process every Nth frame (1=all frames, higher=skip frames for speed)",
    ),
    max_frames: int = Query(
        1000,
        ge=100,
        le=5000,
        description="Maximum frames to process (prevents timeout)",
    ),
):
    """
    Annotate entire video with parking space detection.
    Returns annotated video file with all frames processed.
    
    - **video**: Video file to process
    - **frame_stride**: Process every Nth frame (1=all frames, higher values skip frames for faster processing). Default: 1
    - **max_frames**: Maximum frames to process to prevent timeout on long videos. Default: 1000
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Received annotate-video request: frame_stride={frame_stride}, max_frames={max_frames}")
    
    # Resolve source file (support both 'video' and 'file' field names)
    src = video or file
    if src is None:
        raise HTTPException(
            status_code=400,
            detail="No video file provided. Use field name 'video' (recommended) or 'file' in form-data.",
        )

    # Validate video file
    if not src.filename:
        raise HTTPException(status_code=400, detail="Video filename is required")
    
    logger.info(f"Processing video file: {src.filename}, content_type: {src.content_type}")
    
    # Check content type (allow if not set, as some clients don't send it)
    content_type = src.content_type or ""
    if content_type and not content_type.startswith("video/"):
        logger.warning(f"Unexpected content type: {content_type}, but proceeding anyway")
    
    try:
        data = await src.read()
        if not data or len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty video file")
        logger.info(f"Read video data: {len(data)} bytes")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading video file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading video file: {str(e)}")

    # Use NamedTemporaryFile with delete=False so it persists during processing
    # Then manually delete after reading
    import os
    from pathlib import Path
    
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    tmp_file = None
    try:
        os.close(fd)
        tmp_file = Path(tmp_path)
        tmp_file.write_bytes(data)
        
        try:
            annotated_bytes = annotate_video_file(
                str(tmp_file), 
                frame_stride=frame_stride,
                max_frames=max_frames
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        finally:
            # Clean up temp input file
            if tmp_file and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

        if not annotated_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate annotated video")

        return StreamingResponse(
            io.BytesIO(annotated_bytes),
            media_type="video/mp4",
            headers={
                "Content-Disposition": 'attachment; filename="annotated.mp4"',
                "Content-Length": str(len(annotated_bytes))
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

