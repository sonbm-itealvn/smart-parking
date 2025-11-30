from typing import Dict, Tuple
import io
import numpy as np
import cv2
from PIL import Image

from ..config import SETTINGS


def bytes_to_cv2_image(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_image_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return buf.tobytes()


def centroid_from_xyxy(box_xyxy: np.ndarray) -> Tuple[int, int]:
    return int((box_xyxy[0] + box_xyxy[2]) / 2), int((box_xyxy[1] + box_xyxy[3]) / 2)


def euclid_dist(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    return float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))


def draw_path(
    frame_bgr: np.ndarray,
    start: Tuple[int, int],
    spot_xyxy: np.ndarray,
    landmarks: Dict[int, Tuple[int, int]] | None = None,
) -> np.ndarray:
    # Ensure contiguous array for OpenCV drawing operations
    frame_bgr = frame_bgr.copy()
    landmarks = dict(landmarks or SETTINGS.parking_landmarks)
    landmarks[0] = start
    cx = int((spot_xyxy[0] + spot_xyxy[2]) / 2)
    spot_check = (cx, int(spot_xyxy[1]))

    best_landmark = 0
    min_dist = euclid_dist(spot_check, tuple(landmarks[best_landmark]))

    for key in sorted(landmarks.keys()):
        point = tuple(landmarks[key])
        cv2.circle(frame_bgr, point, 4, (0, 255, 0), -1)
        d = euclid_dist(spot_check, point)
        if d < min_dist:
            min_dist = d
            best_landmark = key

    last_key = 0
    for key in sorted(k for k in landmarks.keys() if k <= best_landmark):
        if key == 0:
            continue
        frame_bgr = cv2.line(
            frame_bgr,
            tuple(landmarks[last_key]),
            tuple(landmarks[key]),
            (0, 255, 0),
            3,
        )
        last_key = key
    frame_bgr = cv2.line(
        frame_bgr,
        tuple(landmarks[best_landmark]),
        spot_check,
        (0, 255, 0),
        3,
    )
    return frame_bgr


def normalize_img_rgb(img_rgb: np.ndarray) -> np.ndarray:
    norm_target = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
    return cv2.normalize(img_rgb, norm_target, 0, 255, cv2.NORM_MINMAX)


