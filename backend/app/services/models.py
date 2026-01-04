from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from typing import TYPE_CHECKING

from ..config import SETTINGS

if TYPE_CHECKING:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_ps_model = None
_lp_model = None
_ocr_processor = None
_ocr_model = None


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


def get_ocr_processor() -> "TrOCRProcessor":
    """Lấy TrOCR processor để preprocess ảnh"""
    global _ocr_processor
    if _ocr_processor is None:
        # Sử dụng model TrOCR cho printed text (tốt cho biển số)
        _ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    return _ocr_processor


def get_ocr_model() -> "VisionEncoderDecoderModel":
    """Lấy TrOCR model để đọc text"""
    global _ocr_model
    if _ocr_model is None:
        # Sử dụng model TrOCR cho printed text (tốt cho biển số)
        _ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        _ocr_model.eval()  # Set to evaluation mode
    return _ocr_model
