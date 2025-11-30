from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2

from backend.app.services.parking_service import recommend_from_frame
from backend.app.services.tracker import CentroidTracker


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "tests"


def _resolve_source(source: str) -> Union[int, str]:
    try:
        return int(source)
    except ValueError:
        return source


def process_stream(
    source: Union[int, str, Path],
    output_dir: Path | None = None,
    output_video: Optional[Path] = None,
    frame_stride: int = 1,
) -> None:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "frame.jpg"
    routed_path = output_dir / "recommendation.jpg"

    capture_source = source if isinstance(source, int) else str(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    writer: Optional[cv2.VideoWriter] = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = CentroidTracker()

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_stride > 1 and (frame_idx % frame_stride) != 0:
                tracker.update([])
                continue
            cv2.imwrite(str(frame_path), frame)
            result = recommend_from_frame(frame, tracker=tracker)
            annotated = result.routed_bgr if result.routed_bgr is not None else result.annotated_bgr
            cv2.imwrite(str(routed_path), annotated)

            if result.routed_bgr is not None:
                print("Recommended parking spot updated.")
            else:
                print(result.message or "All parking spaces are full")

            if output_video:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend parking spaces from a video/image stream.")
    parser.add_argument("source", help="Path to a video file or camera index")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for debug frames")
    parser.add_argument("--output-video", type=Path, help="Path to save annotated recommendation video")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame for speed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve_source(str(args.source))
    process_stream(source, args.output_dir, args.output_video, args.frame_stride)


if __name__ == "__main__":
    main()
