from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import io
import csv
import numpy as np
import cv2
from pathlib import Path

from ..services.models import get_lp_model, get_ocr_reader, LICENSE_LOG_PATH
from ..utils.images import bytes_to_cv2_image, cv2_image_to_png_bytes, normalize_img_rgb


router = APIRouter()


@router.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        data = await image.read()
        frame_bgr = bytes_to_cv2_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    model = get_lp_model()
    reader = get_ocr_reader()

    results = model.predict(frame_bgr, conf=0.6, verbose=False)
    r = results[0]

    plotted = r.plot(conf=False, labels=False)
    annotated_bgr = plotted[..., ::-1]

    texts = []

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        norm_rgb = normalize_img_rgb(crop_rgb)
        result = reader.readtext(norm_rgb)
        if result:
            s = result[0][1]
            texts.append(s)

    png_bytes = cv2_image_to_png_bytes(annotated_bgr)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png", headers={"X-Recognized-Texts": ";".join(texts)})


@router.get("/logs")
def logs():
    logs = []
    if not LICENSE_LOG_PATH.exists():
        return {"logs": logs}
    with LICENSE_LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            plate = row[0]
            ts = row[1] if len(row) > 1 else None
            logs.append({"plate": plate, "timestamp": ts})
    return {"logs": logs}


