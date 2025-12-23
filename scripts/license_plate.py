from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import cv2

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.services.license_plate import detect_license_plates


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "license_plate"


def _resolve_source(source: str) -> Union[int, str]:
    try:
        return int(source)
    except ValueError:
        return source


def process_stream(source: Union[int, str, Path], output_dir: Path | None = None) -> None:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "frame.jpg"
    annotated_path = output_dir / "annotated.jpg"

    capture_source = source if isinstance(source, int) else str(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detect_license_plates(frame, log_results=True)
            if result.texts:
                print("Detected plates:", ", ".join(result.texts))
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(annotated_path), result.annotated_bgr)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run license plate detection on a video/image stream.")
    parser.add_argument("source", help="Path to a video file or camera index")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for debug frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve_source(str(args.source))
    process_stream(source, args.output_dir)


if __name__ == "__main__":
    main()
