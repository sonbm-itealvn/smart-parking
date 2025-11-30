from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import tempfile

import cv2
import numpy as np

from ..config import SETTINGS
from ..utils.images import centroid_from_xyxy, draw_path, euclid_dist
from .models import get_parking_model
from .tracker import CentroidTracker


def _prepare_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame_bgr.shape[:2]
    max_width = SETTINGS.parking_max_width
    if max_width and width > max_width:
        scale = max_width / width
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
        return resized, scale
    return frame_bgr, 1.0


@dataclass
class ParkingRecommendationResult:
    annotated_bgr: np.ndarray
    routed_bgr: Optional[np.ndarray]
    best_box: Optional[np.ndarray]
    message: Optional[str] = None


def recommend_from_frame(
    frame_bgr: np.ndarray,
    tracker: CentroidTracker | None = None,
) -> ParkingRecommendationResult:
    model = get_parking_model()
    processed_frame, scale = _prepare_frame(frame_bgr)
    results = model.predict(processed_frame, conf=SETTINGS.parking_confidence, verbose=False)
    r = results[0]

    plotted = r.plot(conf=False, labels=False)
    annotated_small = plotted[..., ::-1].copy()
    if scale != 1.0:
        annotated_bgr = cv2.resize(
            annotated_small,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        annotated_bgr = annotated_small

    best_box = None
    min_dist = float("inf")
    start = SETTINGS.parking_entry_point

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    if scale != 1.0 and boxes.size > 0:
        boxes = boxes / scale
    classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.empty((0,))

    empty_boxes: list[np.ndarray] = []
    for i, box in enumerate(boxes):
        if int(classes[i]) == 0:
            empty_boxes.append(box)

    tracked_boxes: dict[int, np.ndarray] = {}
    if tracker is not None:
        tracked_boxes = tracker.update(empty_boxes)
        candidate_boxes = list(tracked_boxes.values())
    else:
        candidate_boxes = empty_boxes

    for box in candidate_boxes:
        cx, cy = centroid_from_xyxy(box)
        d = euclid_dist(start, (cx, cy))
        if d < min_dist:
            min_dist = d
            best_box = box

    if tracker is not None and tracked_boxes:
        for object_id, box in tracked_boxes.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated_bgr,
                f"ID {object_id}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    if best_box is None:
        return ParkingRecommendationResult(
            annotated_bgr=annotated_bgr,
            routed_bgr=None,
            best_box=None,
            message="All parking spaces are full",
        )

    routed = draw_path(annotated_bgr, start, best_box)
    return ParkingRecommendationResult(
        annotated_bgr=annotated_bgr,
        routed_bgr=routed,
        best_box=best_box,
    )


def recommend_from_video(
    capture: cv2.VideoCapture,
    *,
    frame_stride: int = 5,
    max_frames: int = 300,
) -> ParkingRecommendationResult:
    frame_idx = 0
    last_frame = None
    tracker = CentroidTracker()

    while frame_idx < max_frames:
        ret, frame_bgr = capture.read()
        if not ret:
            break
        last_frame = frame_bgr
        if frame_stride > 1 and (frame_idx % frame_stride) != 0:
            tracker.update([])
            frame_idx += 1
            continue

        result = recommend_from_frame(frame_bgr, tracker=tracker)
        if result.routed_bgr is not None:
            return result
        frame_idx += 1

    annotated = last_frame if last_frame is not None else np.zeros((1, 1, 3), dtype=np.uint8)
    return ParkingRecommendationResult(
        annotated_bgr=annotated,
        routed_bgr=None,
        best_box=None,
        message="No empty parking space found in video within limits",
    )


def annotate_video_file(video_path: str | Path, *, frame_stride: int = 1) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Failed to open video for annotation")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_path = Path(tmp_out)
    writer: Optional[cv2.VideoWriter] = None
    last_annotated: Optional[np.ndarray] = None
    tracker = CentroidTracker()

    try:
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame = frame_stride <= 1 or (frame_idx % frame_stride) == 0
            if process_frame or last_annotated is None:
                result = recommend_from_frame(frame, tracker=tracker)
                annotated = result.routed_bgr if result.routed_bgr is not None else result.annotated_bgr
                last_annotated = annotated
            else:
                tracker.update([])
                annotated = last_annotated

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            writer.write(annotated)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    data = out_path.read_bytes()
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    return data
