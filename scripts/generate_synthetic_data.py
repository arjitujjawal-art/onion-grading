"""
Synthetic Onion Tray Data Generator
Generates realistic simulated overhead images of onions on a procurement tray,
including a standard ArUco calibration marker (DICT_4X4_50, 25mm physical size).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


def _generate_aruco_marker(marker_id: int = 0, size: int = 100) -> np.ndarray:
    """Generate a standard OpenCV ArUco marker image."""
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
        # Convert to 3-channel BGR
        return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    except Exception:
        # Fallback manual marker generation
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        border = max(4, size // 10)
        cv2.rectangle(img, (0, 0), (size, size), (0, 0, 0), border)
        inner_start = border * 2
        inner_end = size - border * 2
        cv2.rectangle(img, (inner_start, inner_start), (inner_end, inner_end), (255, 255, 255), -1)
        grid_size = 4
        cell_size = (inner_end - inner_start) // grid_size
        binary_str = format(marker_id, "016b")
        for row in range(grid_size):
            for col in range(grid_size):
                bit = int(binary_str[row * grid_size + col])
                val = 0 if bit == 1 else 255
                x = inner_start + col * cell_size
                y = inner_start + row * cell_size
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (val, val, val), -1)
        return img


def generate_tray_image(
    n_onions: int = 8,
    include_calibration: bool = True,
    seed: Optional[int] = 42,
    output_path: Optional[str | Path] = None,
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """
    Generate a synthetic tray image with realistic onions and an optional ArUco calibration marker.
    """
    rng = np.random.default_rng(seed)

    # 1. Background: procurement tray (light textured off-white/beige)
    img = np.full((height, width, 3), (235, 235, 235), dtype=np.uint8)
    # Add subtle texture
    noise = rng.integers(-5, 5, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Tray inner border
    cv2.rectangle(img, (40, 40), (width - 40, height - 40), (190, 190, 190), 3)

    placed_regions: list[Tuple[int, int, int]] = []  # (cx, cy, r)

    # 2. Add ArUco calibration marker (top-left corner)
    if include_calibration:
        marker_size = 100  # pixels representing 25.0 mm -> ~4.0 px/mm
        marker_cx, marker_cy = 100, 100
        marker_img = _generate_aruco_marker(marker_id=0, size=marker_size)
        
        # Paste marker with white card border
        card_size = marker_size + 24
        x1 = marker_cx - card_size // 2
        y1 = marker_cy - card_size // 2
        cv2.rectangle(img, (x1, y1), (x1 + card_size, y1 + card_size), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x1 + card_size, y1 + card_size), (180, 180, 180), 1)

        mx1 = marker_cx - marker_size // 2
        my1 = marker_cy - marker_size // 2
        img[my1 : my1 + marker_size, mx1 : mx1 + marker_size] = marker_img
        placed_regions.append((marker_cx, marker_cy, card_size // 2 + 25))

    # 3. Defect types to distribute
    defect_types = ["healthy", "healthy", "damaged", "sprouted", "rotten"]

    # 4. Generate onions
    attempts = 0
    while len(placed_regions) - (1 if include_calibration else 0) < n_onions and attempts < 200:
        attempts += 1
        # Physical radius in px (approx 35mm - 65mm at 4 px/mm -> 70px to 130px diameter, r = 35 to 65)
        r = int(rng.integers(38, 70))
        cx = int(rng.integers(r + 60, width - r - 60))
        cy = int(rng.integers(r + 60, height - r - 60))

        # Check collision with already placed items
        overlap = False
        for px, py, pr in placed_regions:
            dist = np.hypot(cx - px, cy - py)
            if dist < (r + pr + 15):
                overlap = True
                break
        if overlap:
            continue

        placed_regions.append((cx, cy, r))
        defect = defect_types[len(placed_regions) % len(defect_types)]

        # Base onion color (red/pink/yellow onion tones in BGR)
        # Red onion outer skin: deep reddish-purple
        base_b = int(rng.integers(70, 110))
        base_g = int(rng.integers(80, 120))
        base_r = int(rng.integers(160, 210))

        # Onion drop shadow for depth
        cv2.circle(img, (cx + 6, cy + 6), r + 2, (180, 180, 180), -1)

        # Draw main onion body with radial gradient
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        # Slight deformation for natural organic onion shape
        axes = (r, int(r * rng.uniform(0.88, 1.02)))
        angle = rng.integers(0, 180)
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (base_b, base_g, base_r), -1)
        # Concentric skin rings
        for ring in range(r - 5, 10, -8):
            cv2.ellipse(img, (cx, cy), (ring, int(ring * 0.95)), angle, 0, 360,
                        (max(0, base_b - 15), max(0, base_g - 15), max(0, base_r - 20)), 2)

        # Apply specific defect traits
        if defect == "damaged":
            # Brown skin tear / peeling patch
            defect_r = int(r * 0.35)
            dx = cx + int(rng.integers(-r // 2, r // 2))
            dy = cy + int(rng.integers(-r // 2, r // 2))
            cv2.circle(img, (dx, dy), defect_r, (40, 80, 140), -1)  # brown patch in BGR
        elif defect == "rotten":
            # Dark necrotic sunken spot
            defect_r = int(r * 0.4)
            dx = cx + int(rng.integers(-r // 3, r // 3))
            dy = cy + int(rng.integers(-r // 3, r // 3))
            cv2.circle(img, (dx, dy), defect_r, (30, 30, 45), -1)  # blackened rot
        elif defect == "sprouted":
            # Green shoot protruding from apical tip
            tip_x = cx + int(r * np.cos(np.radians(angle)))
            tip_y = cy - int(r * np.sin(np.radians(angle)))
            cv2.line(img, (tip_x, tip_y), (tip_x + 12, tip_y - 25), (40, 180, 60), 4)
            cv2.line(img, (tip_x + 12, tip_y - 25), (tip_x + 18, tip_y - 40), (50, 200, 70), 3)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)

    return img
