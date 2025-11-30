from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from ..config import SETTINGS
from ..utils.images import normalize_img_rgb
from .license_logs import append_plate
from .models import get_lp_model, get_ocr_reader


@dataclass
class LicensePlateDetectionResult:
    annotated_bgr: np.ndarray
    texts: List[str] = field(default_factory=list)


def detect_license_plates(frame_bgr: np.ndarray, *, log_results: bool = False) -> LicensePlateDetectionResult:
    model = get_lp_model()
    reader = get_ocr_reader()

    results = model.predict(frame_bgr, conf=SETTINGS.license_confidence, verbose=False)
    r = results[0]
    plotted = r.plot(conf=False, labels=False)
    annotated_bgr = plotted[..., ::-1].copy()

    texts: List[str] = []
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
            text = result[0][1]
            texts.append(text)
            if log_results:
                append_plate(text)

    return LicensePlateDetectionResult(annotated_bgr=annotated_bgr, texts=texts)

