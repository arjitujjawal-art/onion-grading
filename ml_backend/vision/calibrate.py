"""Pixel-to-mm calibration using ArUco markers or physical reference objects."""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Standard procurement setup: 25.0 mm ArUco marker (DICT_4X4_50)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_MM = 25.0


def detect_aruco_calibration(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Detect ArUco marker in image.
    Returns (corners_4x2, marker_id) or (None, None).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_list, ids, _ = detector.detectMarkers(gray)

    if corners_list is not None and len(corners_list) > 0 and ids is not None and len(ids) > 0:
        # First detected marker corners reshaped to (4, 2)
        corners = np.array(corners_list[0]).reshape(-1, 2)
        marker_id = int(ids[0][0]) if hasattr(ids[0], "__len__") else int(ids[0])
        return corners, marker_id

    return None, None


def compute_calibration_details(
    image: np.ndarray,
    reference_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> Dict[str, Any]:
    """
    Comprehensive calibration analysis.
    Returns dictionary with calibration status, scale (px/mm), corners, and method.
    """
    corners, marker_id = detect_aruco_calibration(image)
    if corners is not None and len(corners) == 4:
        # Calculate lengths of all 4 edges for robust scale estimation
        edge_lengths = [
            np.linalg.norm(corners[i] - corners[(i + 1) % 4])
            for i in range(4)
        ]
        avg_pixel_side = float(np.mean(edge_lengths))
        if avg_pixel_side > 5.0:
            ppm = avg_pixel_side / reference_mm
            return {
                "calibrated": True,
                "pixels_per_mm": float(ppm),
                "marker_id": marker_id,
                "corners": corners.tolist(),
                "method": "aruco_4x4_50",
                "reference_mm": reference_mm,
                "marker_pixel_size": avg_pixel_side,
            }

    # Fallback template detection
    ppm_fallback = _detect_calibration_card(image, reference_mm)
    if ppm_fallback is not None and ppm_fallback > 0:
        return {
            "calibrated": True,
            "pixels_per_mm": float(ppm_fallback),
            "marker_id": None,
            "corners": None,
            "method": "template_contrast_card",
            "reference_mm": reference_mm,
            "marker_pixel_size": float(ppm_fallback * reference_mm),
        }

    return {
        "calibrated": False,
        "pixels_per_mm": None,
        "marker_id": None,
        "corners": None,
        "method": "none",
        "reference_mm": reference_mm,
        "marker_pixel_size": None,
    }


def compute_pixels_per_mm(
    image: np.ndarray,
    reference_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> Optional[float]:
    """
    Convenience wrapper returning pixels_per_mm or None if uncalibrated.
    """
    details = compute_calibration_details(image, reference_mm)
    return details["pixels_per_mm"] if details["calibrated"] else None


def _generate_template_marker(size: int = 44) -> np.ndarray:
    """Generate a template for the outer black border of a calibration marker."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    border = max(2, size // 10)
    cv2.rectangle(img, (0, 0), (size, size), 0, border)
    cv2.rectangle(img, (2 * border, 2 * border), (size - 2 * border, size - 2 * border), 255, -1)
    return img


def _detect_calibration_card(
    image: np.ndarray,
    reference_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> Optional[float]:
    """Detect high-contrast square calibration reference via multiscale template matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    for template_size in [200, 160, 144, 120, 100, 80]:
        template = _generate_template_marker(template_size)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.94:
            x, y = max_loc
            border = template_size // 10
            interior = gray[y + 2 * border : y + template_size - 2 * border,
                            x + 2 * border : x + template_size - 2 * border]
            if interior.size > 0 and float(np.mean(interior)) > 190:
                return float(template_size / reference_mm)

    return None


def estimate_diameter(onion_mask: np.ndarray, pixels_per_mm: float) -> float:
    """
    Estimate onion physical diameter in mm from its 2D binary segmentation mask.
    Uses equivalent circular diameter: D = 2 * sqrt(Area / pi) / pixels_per_mm.
    """
    if onion_mask is None or onion_mask.size == 0 or pixels_per_mm <= 0:
        return 0.0
    area_pixels = np.sum(onion_mask > 0)
    if area_pixels == 0:
        return 0.0
    diameter_pixels = 2.0 * np.sqrt(area_pixels / np.pi)
    return float(diameter_pixels / pixels_per_mm)
