from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import os
import tempfile

import cv2
import numpy as np

from ..config import SETTINGS
from ..utils.images import centroid_from_xyxy, draw_path, euclid_dist
from .models import get_parking_model
from .tracker import CentroidTracker
from ultralytics import YOLO


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


@dataclass
class VehicleDetection:
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


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


def annotate_video_file(video_path: str | Path, *, frame_stride: int = 1, max_frames: int = 1000) -> bytes:
    """
    Annotate video file with parking space detection.
    
    Args:
        video_path: Path to input video file
        frame_stride: Process every Nth frame (1=all frames)
        max_frames: Maximum number of frames to process (to prevent timeout)
    
    Returns:
        bytes: Annotated video as MP4 bytes
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Failed to open video for annotation")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit processing to prevent timeout
    if total_frames > max_frames:
        # Adjust frame_stride to process max_frames
        adjusted_stride = max(1, total_frames // max_frames)
        if frame_stride < adjusted_stride:
            frame_stride = adjusted_stride

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_path = Path(tmp_out)
    writer: Optional[cv2.VideoWriter] = None
    last_annotated: Optional[np.ndarray] = None
    tracker = CentroidTracker()

    try:
        frame_idx = 0
        processed_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Limit total processed frames
            if processed_count >= max_frames:
                break
                
            process_frame = frame_stride <= 1 or (frame_idx % frame_stride) == 0
            if process_frame or last_annotated is None:
                result = recommend_from_frame(frame, tracker=tracker)
                annotated = result.routed_bgr if result.routed_bgr is not None else result.annotated_bgr
                last_annotated = annotated
                processed_count += 1
            else:
                tracker.update([])
                annotated = last_annotated

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise ValueError("Failed to create video writer")
            
            writer.write(annotated)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ValueError("Failed to generate annotated video")

    data = out_path.read_bytes()
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    return data


def detect_vehicles(frame_bgr: np.ndarray) -> List[VehicleDetection]:
    """
    Detect vehicles and return bounding boxes with confidence.
    This reuses the parking model to detect objects and returns raw boxes.
    """
    model = get_parking_model()
    processed_frame, scale = _prepare_frame(frame_bgr)

    results = model.predict(processed_frame, conf=SETTINGS.parking_confidence, verbose=False)
    r = results[0]

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.zeros(len(boxes))
    class_ids = r.boxes.cls.cpu().numpy() if r.boxes is not None and r.boxes.cls is not None else np.zeros(len(boxes))
    names = getattr(model, "names", {}) or {}

    detections: List[VehicleDetection] = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = float(scores[idx]) if idx < len(scores) else 0.0
        cls_id = int(class_ids[idx]) if idx < len(class_ids) else -1
        cls_name = names.get(cls_id, str(cls_id))

        # Rescale back if the frame was resized
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        detections.append(
            VehicleDetection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            )
        )

    return detections


# Cached pretrained YOLOv8x model (COCO) for fast vehicle detection when requested
_pretrained_vehicle_model: YOLO | None = None


def detect_vehicles_pretrained(frame_bgr: np.ndarray) -> List[VehicleDetection]:
    """
    Detect vehicles using YOLOv8x pretrained on COCO for speed/convenience.
    """
    global _pretrained_vehicle_model
    if _pretrained_vehicle_model is None:
        _pretrained_vehicle_model = YOLO("yolov8x.pt")

    processed_frame, scale = _prepare_frame(frame_bgr)

    results = _pretrained_vehicle_model.predict(processed_frame, conf=0.25, verbose=False)
    r = results[0]

    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.zeros(len(boxes))
    class_ids = r.boxes.cls.cpu().numpy() if r.boxes is not None and r.boxes.cls is not None else np.zeros(len(boxes))
    names = getattr(_pretrained_vehicle_model, "names", {}) or {}

    detections: List[VehicleDetection] = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = float(scores[idx]) if idx < len(scores) else 0.0
        cls_id = int(class_ids[idx]) if idx < len(class_ids) else -1
        cls_name = names.get(cls_id, str(cls_id))

        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        detections.append(
            VehicleDetection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            )
        )

    return detections
