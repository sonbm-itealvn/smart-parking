from ultralytics import YOLO
import easyocr

from ..config import SETTINGS

_ps_model = None
_lp_model = None
_ocr_reader = None


def get_parking_model() -> YOLO:
    global _ps_model
    if _ps_model is None:
        _ps_model = YOLO(str(SETTINGS.parking_model_path))
    return _ps_model


def get_lp_model() -> YOLO:
    global _lp_model
    if _lp_model is None:
        _lp_model = YOLO(str(SETTINGS.license_model_path))
    return _lp_model


def get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"])  # load once
    return _ocr_reader
