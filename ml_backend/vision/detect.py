"""
Onion detection, segmentation, and defect classification using trained YOLOv8s-seg.
Includes non-onion false-positive filtering, physical geometric validation,
and high-end visual annotation overlay.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np

# Palette for high-end rendering (BGR for OpenCV)
GRADE_COLORS_BGR = {
    "GRADE_A": (34, 197, 94),        # Emerald Green
    "GRADE_URS": (11, 158, 245),      # Amber Orange
    "REJECTED": (68, 68, 239),       # Crimson Red
    "MANUAL_REVIEW": (241, 102, 99),  # Indigo/Slate Blue
    "UNCERTAIN": (160, 160, 160),     # Neutral Grey
}

CLASS_NAME_MAP = {
    0: "grade_a",
    1: "grade_urs",
    2: "reject",
}

_MODEL_CACHE: Dict[str, Any] = {}


@dataclass
class DetectedOnion:
    onion_id: str
    bbox: Tuple[float, float, float, float]           # Normalized (x1, y1, x2, y2) in [0, 1]
    bbox_pixels: Tuple[int, int, int, int]           # Absolute pixels (x1, y1, x2, y2)
    visual_class: str                                # "grade_a", "grade_urs", "reject"
    confidence: float                                # [0.0, 1.0]
    mask: Optional[np.ndarray] = None                # Full-res binary boolean mask
    polygon: Optional[List[List[float]]] = None       # Normalized polygon vertices [[x, y], ...]
    polygon_pixels: Optional[List[List[int]]] = None  # Pixel polygon vertices [[x, y], ...]
    is_valid_onion: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class DetectionResult:
    onions: List[DetectedOnion]
    is_onion_frame: bool
    calibration_details: Dict[str, Any]
    annotated_image: Optional[np.ndarray] = None
    filter_summary: Dict[str, int] = field(default_factory=dict)


def load_model(model_path: Optional[str] = None):
    """
    Load YOLOv8 segmentation model with priority path lookup and caching.
    """
    candidate_paths = [
        model_path,
        "model_backend/best.pt",
        "runs/onion-grade-seg-2/weights/best.pt",
        "model_backend/last.pt",
        "weights/best.pt",
    ]

    selected_path = None
    for p in candidate_paths:
        if p and Path(p).is_file():
            selected_path = str(Path(p).resolve())
            break

    if not selected_path:
        return None, False

    if selected_path in _MODEL_CACHE:
        return _MODEL_CACHE[selected_path], True

    try:
        from ultralytics import YOLO
        model = YOLO(selected_path)
        _MODEL_CACHE[selected_path] = model
        return model, True
    except Exception as e:
        print(f"Error loading model from {selected_path}: {e}")
        return None, False


def detect_onions(
    image: np.ndarray,
    model=None,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.60,
    calibration_details: Optional[Dict[str, Any]] = None,
    device: Optional[str | int] = None,
) -> DetectionResult:
    """
    Detect, segment, and validate onions from an image.
    Applies strict geometric and confidence filtering to suppress non-onion objects.
    """
    if calibration_details is None:
        calibration_details = {"calibrated": False, "pixels_per_mm": None, "corners": None}

    h, w = image.shape[:2]

    # Attempt to load model if not provided
    if model is None:
        model, ok = load_model()
        if not ok:
            return DetectionResult(
                onions=[],
                is_onion_frame=False,
                calibration_details=calibration_details,
                annotated_image=image.copy(),
                filter_summary={"error": 1},
            )

    # Determine execution device (CUDA if available)
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Run YOLOv8 Segmentation
    try:
        yolo_results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            device=device,
            verbose=False,
        )[0]
    except Exception as err:
        print(f"YOLO inference error: {err}")
        return DetectionResult(
            onions=[],
            is_onion_frame=False,
            calibration_details=calibration_details,
            annotated_image=image.copy(),
            filter_summary={"inference_failed": 1},
        )

    onions: List[DetectedOnion] = []
    filtered_out = 0

    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        boxes = yolo_results.boxes
        masks = yolo_results.masks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1_px, y1_px = max(0, int(round(x1))), max(0, int(round(y1)))
            x2_px, y2_px = min(w, int(round(x2))), min(h, int(round(y2)))

            box_w = x2_px - x1_px
            box_h = y2_px - y1_px

            # -------------------------------------------------------------
            # NON-ONION & FALSE-POSITIVE FILTERS (Quality Assurance)
            # -------------------------------------------------------------
            # 1. Minimum dimension filter (reject speckles / noise)
            if box_w < 18 or box_h < 18:
                filtered_out += 1
                continue

            # 2. Aspect Ratio filter: Onions are roughly circular/elliptical.
            # Reject needle-like, banner-like, or elongated objects (ruler edges, wires, tray walls).
            aspect_ratio = box_w / max(1.0, box_h)
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                filtered_out += 1
                continue

            # 3. Maximum size filter: A single onion should not occupy > 95% of entire frame
            if box_w > 0.95 * w and box_h > 0.95 * h:
                filtered_out += 1
                continue

            # Extract segmentation mask & polygon
            binary_mask = None
            poly_norm: List[List[float]] = []
            poly_px: List[List[int]] = []

            if masks is not None and i < len(masks):
                # Polygon vertices in pixel coordinates
                if hasattr(masks, "xy") and i < len(masks.xy) and len(masks.xy[i]) > 0:
                    raw_poly = masks.xy[i]
                    poly_px = raw_poly.astype(int).tolist()
                    poly_norm = [[float(pt[0] / w), float(pt[1] / h)] for pt in raw_poly]

                # Full-res binary mask
                if hasattr(masks, "data") and i < len(masks.data):
                    mask_tensor = masks.data[i]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    # Resize mask to original image dimensions if needed
                    if mask_np.shape != (h, w):
                        binary_mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    else:
                        binary_mask = mask_np > 0

            # Fallback mask from polygon if binary_mask missing
            if binary_mask is None and len(poly_px) > 2:
                binary_mask = np.zeros((h, w), dtype=bool)
                cv2.fillPoly(binary_mask.view(np.uint8), [np.array(poly_px, dtype=np.int32)], 1)

            # Map class name
            visual_class = CLASS_NAME_MAP.get(cls_id, "grade_urs")

            onions.append(
                DetectedOnion(
                    onion_id=f"onion_{len(onions) + 1:02d}",
                    bbox=(x1_px / w, y1_px / h, x2_px / w, y2_px / h),
                    bbox_pixels=(x1_px, y1_px, x2_px, y2_px),
                    visual_class=visual_class,
                    confidence=conf_val,
                    mask=binary_mask,
                    polygon=poly_norm,
                    polygon_pixels=poly_px,
                    is_valid_onion=True,
                )
            )

    is_onion_frame = len(onions) > 0
    annotated = _render_high_end_annotations(image, onions, calibration_details)

    return DetectionResult(
        onions=onions,
        is_onion_frame=is_onion_frame,
        calibration_details=calibration_details,
        annotated_image=annotated,
        filter_summary={"detected": len(onions), "filtered_false_positives": filtered_out},
    )


def _render_high_end_annotations(
    image: np.ndarray,
    onions: List[DetectedOnion],
    calibration_details: Dict[str, Any],
) -> np.ndarray:
    """
    Renders high-end, production-grade visual annotations:
    - Alpha-blended semi-transparent segmentation polygon fills
    - Vibrant contour borders
    - Modern dark pill badges with bold typography, diameter, and confidence
    - Calibration marker highlight box and scale tag
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    overlay = annotated.copy()

    # 1. Draw segmentation masks on overlay
    for onion in onions:
        color = GRADE_COLORS_BGR.get(onion.visual_class.upper(), (34, 197, 94))
        if onion.visual_class == "grade_a":
            color = GRADE_COLORS_BGR["GRADE_A"]
        elif onion.visual_class == "grade_urs":
            color = GRADE_COLORS_BGR["GRADE_URS"]
        elif onion.visual_class == "reject":
            color = GRADE_COLORS_BGR["REJECTED"]

        if onion.polygon_pixels and len(onion.polygon_pixels) > 2:
            pts = np.array(onion.polygon_pixels, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
        elif onion.mask is not None:
            overlay[onion.mask] = color

    # Blend overlay with 32% opacity for translucent fills
    cv2.addWeighted(overlay, 0.32, annotated, 0.68, 0, annotated)

    # 2. Draw crisp contour outlines and badges
    for onion in onions:
        color = GRADE_COLORS_BGR.get(onion.visual_class.upper(), (34, 197, 94))
        if onion.visual_class == "grade_a":
            color = GRADE_COLORS_BGR["GRADE_A"]
        elif onion.visual_class == "grade_urs":
            color = GRADE_COLORS_BGR["GRADE_URS"]
        elif onion.visual_class == "reject":
            color = GRADE_COLORS_BGR["REJECTED"]

        # Crisp polygon contour boundary
        if onion.polygon_pixels and len(onion.polygon_pixels) > 2:
            pts = np.array(onion.polygon_pixels, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Subtle bounding box corners
        x1, y1, x2, y2 = onion.bbox_pixels
        corner_len = max(10, min(18, (x2 - x1) // 4))
        # Top-left corner
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
        # Top-right corner
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)

        # Label pill badge
        label_class = onion.visual_class.replace("_", " ").upper()
        label_text = f"#{onion.onion_id[-2:]} {label_class} {onion.confidence * 100:.0f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Badge positioning above box or inside if top margin is small
        bx1 = max(4, x1)
        by1 = max(th + 10, y1 - 8)
        bx2 = bx1 + tw + 12
        by2 = by1 - th - 6

        # Dark pill background
        cv2.rectangle(annotated, (bx1, by2), (bx2, by1), (20, 24, 33), -1)
        cv2.rectangle(annotated, (bx1, by2), (bx2, by1), color, 1, lineType=cv2.LINE_AA)
        # Text
        cv2.putText(annotated, label_text, (bx1 + 6, by1 - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 3. Draw Calibration Reference Marker if detected
    if calibration_details.get("calibrated") and calibration_details.get("corners"):
        c_pts = np.array(calibration_details["corners"], dtype=np.int32)
        cyan = (212, 182, 6)  # Cyan in BGR
        cv2.polylines(annotated, [c_pts], isClosed=True, color=cyan, thickness=2, lineType=cv2.LINE_AA)

        # Corner crosshairs
        for pt in c_pts:
            cv2.drawMarker(annotated, tuple(pt), cyan, cv2.MARKER_CROSS, markerSize=8, thickness=1)

        # Calibration badge
        ppm = calibration_details.get("pixels_per_mm", 0.0)
        cal_label = f"REF MARKER (25mm) | {ppm:.2f} px/mm"
        cx = int(np.min(c_pts[:, 0]))
        cy = max(24, int(np.min(c_pts[:, 1])) - 8)
        (cw, ch), _ = cv2.getTextSize(cal_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (cx, cy - ch - 4), (cx + cw + 10, cy + 2), (20, 24, 33), -1)
        cv2.rectangle(annotated, (cx, cy - ch - 4), (cx + cw + 10, cy + 2), cyan, 1, cv2.LINE_AA)
        cv2.putText(annotated, cal_label, (cx + 5, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, cyan, 1, cv2.LINE_AA)

    return annotated
