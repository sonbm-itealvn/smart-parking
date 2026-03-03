from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import base64
import re

import cv2
import numpy as np
from PIL import Image
import torch
from openai import OpenAI

from ..config import SETTINGS
from ..utils.images import preprocess_license_plate_for_ocr
from .license_logs import append_plate
from .models import get_lp_model, get_ocr_processor, get_ocr_model


@dataclass
class LicensePlateInfo:
    """Thông tin chi tiết về một biển số được phát hiện"""
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    detection_confidence: float  # Confidence từ YOLO model


@dataclass
class LicensePlateDetectionResult:
    annotated_bgr: np.ndarray
    texts: List[str] = field(default_factory=list)
    details: List[LicensePlateInfo] = field(default_factory=list)


def _preprocess_plate_image(img_bgr: np.ndarray) -> List[tuple[str, np.ndarray, bool]]:
    """
    Tạo nhiều phiên bản preprocessed của ảnh biển số để thử OCR.
    Trả về danh sách các tuple: (method_name, processed_image, is_split)
    is_split=True nghĩa là ảnh đã được chia thành 2 phần (trên/dưới)
    """
    processed_images = []
    h, w = img_bgr.shape[:2]
    
    # Resize lớn hơn để TrOCR đọc được cả 2 dòng (tối thiểu 400px width, 200px height)
    min_width = 400
    min_height = 200
    scale_w = min_width / w if w < min_width else 1.0
    scale_h = min_height / h if h < min_height else 1.0
    scale = max(scale_w, scale_h)
    
    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]
    
    # Method 1: Original RGB - toàn bộ ảnh
    crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    processed_images.append(("original", crop_rgb, False))
    
    # Method 2: Chia ảnh thành 2 phần (trên/dưới) để đọc riêng
    # Phần trên (dòng đầu tiên)
    top_half = img_bgr[:h//2, :]
    if top_half.size > 0:
        top_rgb = cv2.cvtColor(top_half, cv2.COLOR_BGR2RGB)
        processed_images.append(("top_half", top_rgb, True))
    
    # Phần dưới (dòng thứ hai)
    bottom_half = img_bgr[h//2:, :]
    if bottom_half.size > 0:
        bottom_rgb = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2RGB)
        processed_images.append(("bottom_half", bottom_rgb, True))
    
    # Method 3: CLAHE enhanced (tốt cho ảnh tối) - toàn bộ ảnh
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    processed_images.append(("clahe", enhanced_rgb, False))
    
    # Method 4: CLAHE trên phần trên
    if top_half.size > 0:
        if len(top_half.shape) == 3:
            top_gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
        else:
            top_gray = top_half.copy()
        top_enhanced = clahe.apply(top_gray)
        top_enhanced_rgb = cv2.cvtColor(top_enhanced, cv2.COLOR_GRAY2RGB)
        processed_images.append(("top_clahe", top_enhanced_rgb, True))
    
    # Method 5: CLAHE trên phần dưới
    if bottom_half.size > 0:
        if len(bottom_half.shape) == 3:
            bottom_gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        else:
            bottom_gray = bottom_half.copy()
        bottom_enhanced = clahe.apply(bottom_gray)
        bottom_enhanced_rgb = cv2.cvtColor(bottom_enhanced, cv2.COLOR_GRAY2RGB)
        processed_images.append(("bottom_clahe", bottom_enhanced_rgb, True))
    
    # Method 6: Threshold binary (tốt cho ảnh có độ tương phản cao)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    processed_images.append(("binary", binary_rgb, False))
    
    # Method 7: Adaptive threshold (tốt cho ảnh có ánh sáng không đều)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    adaptive_rgb = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
    processed_images.append(("adaptive", adaptive_rgb, False))
    
    return processed_images


def _validate_and_format_plate_text(text: str) -> str:
    """
    Validate và format biển số theo chuẩn Việt Nam.
    Format: XX-YZ NNNN hoặc XXYZ-NNNN hoặc XXG NNN.NN (như 30G 535.07)
    """
    # Loại bỏ ký tự không hợp lệ (giữ dấu chấm cho số thập phân)
    text = re.sub(r'[^A-Z0-9\s\-.]', '', text.upper())
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    
    # Nếu có gạch ngang, giữ nguyên format
    if '-' in text:
        return text.strip()
    
    # Nếu không có gạch ngang, thử format lại
    # Pattern: số-số-chữ-số hoặc số-số-chữ-chữ-số hoặc số-chữ số.số
    # Ví dụ: "66L6 6789" -> "66-L6 6789"
    # Ví dụ: "30G 535.07" -> giữ nguyên
    parts = text.split()
    if len(parts) >= 2:
        # Phần đầu thường là số-số-chữ-số hoặc số-chữ
        first_part = parts[0]
        # Tìm vị trí chữ cái đầu tiên
        for i, char in enumerate(first_part):
            if char.isalpha():
                if i > 0:
                    # Chèn gạch ngang trước chữ cái nếu chưa có
                    if '-' not in first_part:
                        formatted = first_part[:i] + '-' + first_part[i:] + ' ' + ' '.join(parts[1:])
                        return formatted.strip()
                break
    
    return text.strip()


def _is_valid_plate_text(text: str) -> bool:
    """
    Kiểm tra xem text có giống biển số xe Việt Nam không.
    Biển số VN thường có: số-số-chữ-số hoặc số-chữ số.số
    """
    if not text or len(text.strip()) < 2:
        return False
    
    # Loại bỏ khoảng trắng và dấu gạch ngang để kiểm tra
    clean_text = re.sub(r'[\s\-.]', '', text.upper())
    
    # Phải có ít nhất 1 chữ số và có thể có chữ cái
    if not re.search(r'[0-9]', clean_text):
        return False
    
    # Độ dài hợp lý cho biển số:
    # - Biển đầy đủ: từ 4 đến 12 ký tự sau khi loại bỏ khoảng trắng
    # - Hàng trên của biển 2 dòng (ví dụ: "30G"): 3 ký tự, có cả số và chữ
    clean_len = len(clean_text)
    if clean_len > 12:
        return False
    if clean_len < 3:
        return False
    if clean_len == 3:
        # Chấp nhận chuỗi 3 ký tự có cả số và chữ cái
        if not (re.search(r"[0-9]", clean_text) and re.search(r"[A-Z]", clean_text)):
            return False
    
    # Không được có quá nhiều ký tự đặc biệt
    special_chars = len(re.findall(r'[^A-Z0-9]', text))
    if special_chars > len(text) * 0.3:  # Không quá 30% là ký tự đặc biệt
        return False
    
    # Phải có ít nhất một chữ số liên tiếp (ít nhất 2 số)
    if not re.search(r'[0-9]{2,}', clean_text):
        return False
    
    return True


_PLATE_PATTERN = re.compile(r"(\d{2,3}[A-Z]-?\d{5})")


def _format_plate_standard(plate: str) -> str:
    """
    Chuẩn hóa về dạng chuẩn: 30G-12345
    - 2–3 số đầu
    - 1 chữ cái
    - 5 số cuối
    """
    cleaned = plate.upper().replace(" ", "").replace(".", "").replace("_", "")
    match = _PLATE_PATTERN.search(cleaned)
    if not match:
        return ""
    token = match.group(1).replace("-", "")
    prefix = token[:-5]
    digits = token[-5:]
    return f"{prefix}-{digits}"


_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _run_gpt_ocr(img_bgr: np.ndarray) -> tuple[str, float]:
    """
    Sử dụng OpenAI GPT-4.1-mini (vision) để đọc biển số.
    Trả về chuỗi duy nhất dạng 30G-12345 nếu đọc được, ngược lại trả ("", 0.0).
    """
    try:
        # Chuyển ảnh sang PNG base64
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        success, buf = cv2.imencode(".png", img_rgb)
        if not success:
            return ("", 0.0)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        client = _get_openai_client()
        prompt = (
            "Bạn là hệ thống OCR chuyên đọc biển số xe Việt Nam.\n"
            "Hãy nhìn vào ảnh và trả về CHỈ MỘT chuỗi biển số theo đúng định dạng:\n"
            "- 2 hoặc 3 chữ số, sau đó 1 chữ cái in hoa, sau đó dấu gạch ngang '-', sau đó 5 chữ số.\n"
            "Ví dụ: 30G-49344\n"
            "Không trả lời gì khác ngoài chuỗi biển số (không giải thích, không xuống dòng)."
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                        },
                    ],
                }
            ],
        )

        # Lấy text đầu ra
        raw_text = ""
        if response.output and response.output[0].content:
            raw_text = response.output[0].content[0].text or ""
        candidate = (raw_text or "").strip().upper()
        plate = _format_plate_standard(candidate)

        # Log debug ra console để dễ kiểm tra
        print(
            "[GPT_OCR_DEBUG] raw_text=", repr(raw_text),
            "| candidate=", repr(candidate),
            "| formatted_plate=", repr(plate),
            flush=True,
        )

        if not plate:
            return ("", 0.0)
        return (plate, 0.9)
    except Exception as e:
        # Nếu có lỗi (API, network...) thì coi như không đọc được
        print("[GPT_OCR_ERROR]", repr(e), flush=True)
        return ("", 0.0)


def _normalize_plate_pattern(text: str) -> str:
    """
    Chuẩn hóa chuỗi OCR về dạng biển số có:
    - 2–3 số đầu
    - 1 chữ cái
    - 4–5 số cuối

    Nếu bắt được pattern này thì trả đúng phần khớp (ví dụ: "30G 493.44" -> "30G49344").
    Nếu không bắt được (ví dụ chỉ đọc được "493.64") thì trả về chuỗi đã làm sạch
    (giữ nguyên các số để không mất dữ liệu).
    """
    # Giữ lại chữ và số, bỏ ký tự khác (khoảng trắng, chấm, gạch...)
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not cleaned:
        return ""

    # Cố gắng tìm đúng mẫu: 2–3 số, 1 chữ cái, 4–5 số
    match = re.search(r"(\d{2,3}[A-Z]\d{4,5})", cleaned)
    if match:
        return match.group(1)

    # Nếu không khớp pattern có chữ cái, vẫn trả cleaned (chỉ số hoặc số lẫn chữ)
    return cleaned


def _run_trocr_ocr(img_rgb: np.ndarray, processor, model) -> tuple[str, float]:
    """
    Hàm cũ dùng TrOCR – giữ lại để backward compatibility nếu cần.
    Hiện tại pipeline đã chuyển sang dùng GPT, nên hàm này không còn được gọi.
    """
    return ("", 0.0)


def detect_license_plates(frame_bgr: np.ndarray, *, log_results: bool = False) -> LicensePlateDetectionResult:
    model = get_lp_model()
    # Giữ lại để tương thích, nhưng OCR chính dùng GPT
    processor = get_ocr_processor()
    ocr_model = get_ocr_model()

    results = model.predict(frame_bgr, conf=SETTINGS.license_confidence, verbose=False)
    r = results[0]
    plotted = r.plot(conf=False, labels=False)
    annotated_bgr = plotted[..., ::-1].copy()

    texts: List[str] = []
    details: List[LicensePlateInfo] = []
    
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
    confidences = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.zeros(len(boxes))
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        
        detection_conf = float(confidences[idx]) if idx < len(confidences) else 0.0
        
        # Dùng GPT-4.1-mini để đọc trực tiếp crop biển số
        plate_text, ocr_confidence = _run_gpt_ocr(crop_bgr)
        if plate_text:
            texts.append(plate_text)
            details.append(
                LicensePlateInfo(
                    text=plate_text,
                    confidence=ocr_confidence,
                    bbox=[x1, y1, x2, y2],
                    detection_confidence=detection_conf
                )
            )
            
            # Vẽ text lên ảnh để dễ kiểm tra
            label = f"{plate_text} ({ocr_confidence:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_bgr,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated_bgr,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            if log_results:
                append_plate(plate_text)

    return LicensePlateDetectionResult(annotated_bgr=annotated_bgr, texts=texts, details=details)

