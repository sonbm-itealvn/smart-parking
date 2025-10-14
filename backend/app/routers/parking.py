from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import numpy as np
import cv2

from ..services.models import get_parking_model
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes, centroid_from_xyxy, euclid_dist, draw_path


router = APIRouter()


@router.post("/recommend")
async def recommend(image: UploadFile = File(...)):
    try:
        data = await image.read()
        frame_bgr = bytes_to_cv2_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    model = get_parking_model()

    results = model.predict(frame_bgr, conf=0.6, verbose=False)
    r = results[0]

    plotted = r.plot(conf=False, labels=False)
    annotated_bgr = plotted[..., ::-1]

    best_box = None
    min_dist = float("inf")
    start = (110, 5)

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.empty((0,))

    for i, box in enumerate(boxes):
        if int(classes[i]) == 0:
            cx, cy = centroid_from_xyxy(box)
            d = euclid_dist(start, (cx, cy))
            if d < min_dist:
                min_dist = d
                best_box = box

    if best_box is None:
        return JSONResponse({"message": "All parking spaces are full"})

    routed = draw_path(annotated_bgr, start, best_box)
    png_bytes = cv2_image_to_png_bytes(routed)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


