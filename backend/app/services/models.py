from pathlib import Path
from ultralytics import YOLO
import easyocr


REPO_ROOT = Path(__file__).resolve().parents[3]
PS_MODEL_PATH = REPO_ROOT / "ParkingSpace" / "models" / "best.pt"
LP_MODEL_PATH = REPO_ROOT / "LicensePlate" / "models" / "best.pt"
LICENSE_LOG_PATH = REPO_ROOT / "LicensePlate" / "log.csv"

_ps_model = None
_lp_model = None
_ocr_reader = None


def get_parking_model() -> YOLO:
    global _ps_model
    if _ps_model is None:
        _ps_model = YOLO(str(PS_MODEL_PATH))
    return _ps_model


def get_lp_model() -> YOLO:
    global _lp_model
    if _lp_model is None:
        _lp_model = YOLO(str(LP_MODEL_PATH))
    return _lp_model


def get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"])  # load once
    return _ocr_reader


