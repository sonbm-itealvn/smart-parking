from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import re

import cv2
import numpy as np
from PIL import Image
import torch

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
    
    # Độ dài hợp lý cho biển số (từ 4 đến 12 ký tự sau khi loại bỏ khoảng trắng)
    if len(clean_text) < 4 or len(clean_text) > 12:
        return False
    
    # Không được có quá nhiều ký tự đặc biệt
    special_chars = len(re.findall(r'[^A-Z0-9]', text))
    if special_chars > len(text) * 0.3:  # Không quá 30% là ký tự đặc biệt
        return False
    
    # Phải có ít nhất một chữ số liên tiếp (ít nhất 2 số)
    if not re.search(r'[0-9]{2,}', clean_text):
        return False
    
    return True


def _run_trocr_ocr(img_rgb: np.ndarray, processor, model) -> tuple[str, float]:
    """
    Chạy TrOCR trên ảnh và trả về (text, confidence).
    TrOCR không trả về confidence trực tiếp, nên ta dùng heuristic.
    """
    try:
        # Convert numpy array to PIL Image
        # Đảm bảo ảnh có kích thước hợp lý
        h, w = img_rgb.shape[:2]
        if h < 10 or w < 10:
            return ("", 0.0)
        
        pil_image = Image.fromarray(img_rgb)
        
        # Preprocess với processor
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        
        # Generate text với max_length hợp lý cho biển số
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=20,  # Giới hạn độ dài cho biển số
                num_beams=5,   # Beam search để có kết quả tốt hơn
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Làm sạch text
        text = generated_text.strip()
        
        # Kiểm tra tính hợp lệ của text
        if not _is_valid_plate_text(text):
            return ("", 0.0)
        
        # Tính confidence dựa trên format và độ dài
        confidence = 0.5
        
        # Tăng confidence nếu có format giống biển số
        if re.search(r'[0-9]', text) and re.search(r'[A-Z]', text):
            confidence += 0.2
        
        # Tăng confidence nếu có dấu gạch ngang hoặc khoảng trắng (format chuẩn)
        if '-' in text or ' ' in text:
            confidence += 0.1
        
        # Tăng confidence nếu độ dài hợp lý (6-10 ký tự)
        if 6 <= len(text.replace(' ', '').replace('-', '')) <= 10:
            confidence += 0.1
        
        confidence = min(0.95, confidence)
        
        return (text, confidence)
    except Exception as e:
        # Log lỗi để debug (có thể bỏ sau)
        # print(f"TrOCR error: {e}")
        return ("", 0.0)


def detect_license_plates(frame_bgr: np.ndarray, *, log_results: bool = False) -> LicensePlateDetectionResult:
    model = get_lp_model()
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
        
        # Thử nhiều phương pháp preprocessing
        processed_images = _preprocess_plate_image(crop_bgr)
        
        best_result = None
        best_confidence = 0.0
        best_text = ""
        
        # Lưu kết quả từ các phần chia (top/bottom) để combine sau
        top_results = []
        bottom_results = []
        full_results = []
        
        # Thử OCR trên tất cả các phiên bản đã preprocess
        for method_name, processed_img, is_split in processed_images:
            try:
                # Chạy TrOCR
                ocr_text_raw, ocr_confidence = _run_trocr_ocr(processed_img, processor, ocr_model)
                
                if not ocr_text_raw or len(ocr_text_raw.strip()) < 2:
                    continue
                
                # Lọc text (giữ dấu chấm vì có thể là "535.07")
                ocr_text_clean = re.sub(r'[^A-Z0-9\s\-.]', '', ocr_text_raw.upper())
                ocr_text_clean = re.sub(r'\s+', ' ', ocr_text_clean).strip()
                
                # Kiểm tra lại tính hợp lệ sau khi lọc
                if not _is_valid_plate_text(ocr_text_clean):
                    # Debug: log text không hợp lệ (có thể bỏ sau)
                    # print(f"Invalid plate text from {method_name}: '{ocr_text_raw}' -> '{ocr_text_clean}'")
                    continue
                
                if len(ocr_text_clean.strip()) >= 2:
                    # Phân loại kết quả
                    if is_split:
                        if 'top' in method_name:
                            top_results.append((ocr_text_clean, ocr_confidence))
                        elif 'bottom' in method_name:
                            bottom_results.append((ocr_text_clean, ocr_confidence))
                    else:
                        full_results.append((ocr_text_clean, ocr_confidence))
                        
            except Exception as e:
                # Bỏ qua lỗi và thử method tiếp theo
                # print(f"Error in OCR method {method_name}: {e}")
                continue
        
        # Combine kết quả từ top và bottom nếu có
        if top_results and bottom_results:
            # Lấy kết quả tốt nhất từ mỗi phần
            top_text, top_conf = max(top_results, key=lambda x: (x[1], len(x[0])))
            bottom_text, bottom_conf = max(bottom_results, key=lambda x: (x[1], len(x[0])))
            
            # Combine: top + bottom
            combined_text = f"{top_text} {bottom_text}".strip()
            combined_confidence = (top_conf + bottom_conf) / 2.0
            
            # Format lại
            combined_text = _validate_and_format_plate_text(combined_text)
            
            # Kiểm tra lại tính hợp lệ sau khi combine
            if _is_valid_plate_text(combined_text) and len(combined_text.strip()) >= 2:
                full_results.append((combined_text, combined_confidence))
        
        # Chọn kết quả tốt nhất từ tất cả các phương pháp
        if full_results:
            # Ưu tiên kết quả có confidence cao nhất, sau đó là độ dài
            # Nhưng chỉ chọn kết quả hợp lệ
            valid_results = [(t, c) for t, c in full_results if _is_valid_plate_text(t)]
            if valid_results:
                best_result = max(valid_results, key=lambda x: (x[1], len(x[0])))
                best_text, best_confidence = best_result
            else:
                best_result = None
        
        # Sử dụng kết quả tốt nhất
        if best_result and len(best_text.strip()) >= 2:
            ocr_text, ocr_confidence = best_result
            
            texts.append(ocr_text)
            details.append(
                LicensePlateInfo(
                    text=ocr_text,
                    confidence=ocr_confidence,
                    bbox=[x1, y1, x2, y2],
                    detection_confidence=detection_conf
                )
            )
            
            # Vẽ text lên ảnh để dễ kiểm tra
            label = f"{ocr_text} ({ocr_confidence:.2f})"
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
                append_plate(ocr_text)

    return LicensePlateDetectionResult(annotated_bgr=annotated_bgr, texts=texts, details=details)

