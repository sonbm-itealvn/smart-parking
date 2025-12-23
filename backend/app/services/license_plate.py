from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import re

import cv2
import numpy as np

from ..config import SETTINGS
from ..utils.images import preprocess_license_plate_for_ocr
from .license_logs import append_plate
from .models import get_lp_model, get_ocr_reader


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


def detect_license_plates(frame_bgr: np.ndarray, *, log_results: bool = False) -> LicensePlateDetectionResult:
    model = get_lp_model()
    reader = get_ocr_reader()

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
        
        # Thử nhiều phương pháp preprocessing để có kết quả tốt nhất
        # Method 1: Preprocessed image (CLAHE enhanced)
        processed_rgb = preprocess_license_plate_for_ocr(crop_bgr)
        ocr_results_processed = reader.readtext(processed_rgb, detail=1)
        
        # Method 2: Original image (convert BGR to RGB)
        crop_rgb_original = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        ocr_results_original = reader.readtext(crop_rgb_original, detail=1)
        
        # Chọn phương pháp cho nhiều kết quả hơn (thường là original tốt hơn cho biển số đầy đủ)
        all_results = []
        if ocr_results_original:
            all_results = ocr_results_original
        elif ocr_results_processed:
            all_results = ocr_results_processed
        
        if all_results:
            # Combine tất cả các text detection từ cùng một biển số
            # EasyOCR trả về: [[bbox, text, confidence], ...]
            # Bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
            # Sắp xếp các detection theo vị trí (từ trái sang phải, từ trên xuống dưới)
            def get_sort_key(result):
                """Lấy key để sắp xếp: ưu tiên y (dòng), sau đó x (cột)"""
                if len(result) < 1 or not result[0]:
                    return (0, 0)
                bbox = result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Lấy y trung bình (dòng) và x trung bình (cột)
                y_avg = sum(point[1] for point in bbox) / len(bbox)
                x_avg = sum(point[0] for point in bbox) / len(bbox)
                return (y_avg, x_avg)
            
            sorted_results = sorted(all_results, key=get_sort_key)
            
            # Combine các text lại với nhau
            combined_texts = []
            total_confidence = 0.0
            count = 0
            
            for result in sorted_results:
                if len(result) < 2:
                    continue
                text = result[1].strip()
                conf = float(result[2]) if len(result) > 2 else 0.0
                
                # Lọc text
                text = re.sub(r'[^A-Z0-9\s\-]', '', text.upper())
                if len(text.strip()) > 0:
                    combined_texts.append(text)
                    total_confidence += conf
                    count += 1
            
            if combined_texts:
                # Kết hợp các text lại
                # Biển số VN thường có format: "66-L6 6789" hoặc "66L6-6789"
                # Có thể có khoảng trắng hoặc gạch ngang giữa các phần
                
                # Thử combine với khoảng trắng trước
                ocr_text = ' '.join(combined_texts)
                
                # Nếu có gạch ngang trong text, giữ nguyên format
                if '-' in ocr_text:
                    # Format có gạch ngang: "66-L6 6789" -> giữ nguyên
                    ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
                else:
                    # Không có gạch ngang, thử thêm gạch ngang nếu phù hợp format
                    # Ví dụ: "66L6 6789" -> "66-L6 6789" hoặc "66L6-6789"
                    # Pattern: số-chữ-số hoặc số-số-chữ-số
                    # Tạm thời giữ nguyên với khoảng trắng
                    ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
                
                # Loại bỏ khoảng trắng thừa nhưng giữ khoảng trắng giữa các phần chính
                # Ví dụ: "66-L6 6789" -> giữ nguyên
                ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
                
                ocr_confidence = total_confidence / count if count > 0 else 0.0
                
                # Bỏ qua nếu text quá ngắn
                if len(ocr_text.strip()) >= 2:
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
                    # Vẽ background cho text
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
                    # Vẽ text
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

