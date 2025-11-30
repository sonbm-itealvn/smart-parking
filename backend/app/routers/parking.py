from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import cv2
import tempfile

from ..services.parking_service import recommend_from_frame, recommend_from_video, annotate_video_file
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes


router = APIRouter()


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
async def annotate_video(video: UploadFile = File(...), frame_stride: int = 1):
    try:
        data = await video.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        try:
            annotated_bytes = annotate_video_file(tmp.name, frame_stride=frame_stride)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return StreamingResponse(
        io.BytesIO(annotated_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="annotated.mp4"'},
    )

