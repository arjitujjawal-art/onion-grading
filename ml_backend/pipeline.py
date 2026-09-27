"""
Unified Onion Grading Pipeline
Orchestrates:
Image Input → Metric Calibration → YOLOv8s-Seg Detection & Filtering →
Physical Sizing → Policy Rule Grading Engine → Digital Evidence Report Generation
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import json
import cv2
import numpy as np

from ml_backend.vision.calibrate import compute_calibration_details, estimate_diameter
from ml_backend.vision.detect import detect_onions, load_model
from ml_backend.vision.grade import (
    GradingPolicy,
    OnionGradingResult,
    grade_onion,
    grade_batch,
    load_policy,
)
from ml_backend.report import generate_report


def process_image(
    image_input: Union[str, Path, np.ndarray],
    policy_input: Union[str, Path, Dict[str, Any], GradingPolicy] = "shared/policies/default_policy.json",
    batch_id: str = "BATCH-001",
    farmer_id: str = "FARMER-001",
    officer_id: str = "OFFICER-001",
    centre: str = "Procurement Centre A",
    gps: Optional[Tuple[float, float]] = None,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.35,
    generate_artifacts: bool = True,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Execute the unified onion grading pipeline.

    Args:
        image_input: File path or pre-loaded BGR/RGB numpy array.
        policy_input: Path to policy JSON, policy dict, or GradingPolicy instance.
        batch_id: Procurement batch identifier.
        farmer_id: Farmer identification code.
        officer_id: Inspecting procurement officer code.
        centre: Name of mandi/procurement centre.
        gps: Optional (lat, lon) coordinates.
        model_path: Optional custom weights path.
        conf_threshold: Confidence threshold for YOLO detection.
        generate_artifacts: Whether to write HTML/PDF/QR files to output_dir.
        output_dir: Directory where report artifacts are saved.

    Returns:
        Structured pipeline result dictionary.
    """
    # 1. Load image
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError("image_input must be a file path or numpy array.")

    # 2. Resolve Policy
    if isinstance(policy_input, GradingPolicy):
        policy = policy_input
        policy_dict = policy.to_dict()
    elif isinstance(policy_input, dict):
        policy = GradingPolicy(**policy_input)
        policy_dict = policy_input
    elif isinstance(policy_input, (str, Path)):
        policy = load_policy(policy_input)
        policy_dict = policy.to_dict()
    else:
        policy = GradingPolicy()
        policy_dict = policy.to_dict()

    # 3. Calibration: Detect physical reference (ArUco 25mm marker)
    cal_details = compute_calibration_details(img_bgr)
    calibration_ok = cal_details["calibrated"]
    pixels_per_mm = cal_details["pixels_per_mm"]

    # 4. Load Model & Run Detection with False Positive Filtering
    model, model_loaded = load_model(model_path)
    detection = detect_onions(
        image=img_bgr,
        model=model,
        conf_threshold=conf_threshold,
        calibration_details=cal_details,
    )

    # 5. Evaluate Physical Sizing and Policy Rules
    results: List[OnionGradingResult] = []
    for onion in detection.onions:
        # Calculate diameter in mm
        if calibration_ok and pixels_per_mm is not None and pixels_per_mm > 0:
            if onion.mask is not None:
                diameter_mm = estimate_diameter(onion.mask, pixels_per_mm)
            else:
                # Bbox width as fallback
                x1, y1, x2, y2 = onion.bbox_pixels
                diameter_mm = (x2 - x1) / pixels_per_mm
            is_calibrated = True
        else:
            # Fallback size estimation (assuming nominal camera distance if uncalibrated)
            # Estimate nominal 45mm for typical full-frame view, or proportional to bbox
            x1, y1, x2, y2 = onion.bbox_pixels
            nominal_ppm = 2.5
            diameter_mm = (x2 - x1) / nominal_ppm
            is_calibrated = False

        # Apply Policy Rule Engine
        final_grade, reason_codes, defect_flags = grade_onion(
            diameter_mm=diameter_mm,
            visual_class=onion.visual_class,
            confidence=onion.confidence,
            calibration_ok=is_calibrated,
            policy=policy,
        )

        results.append(
            OnionGradingResult(
                onion_id=onion.onion_id,
                bbox=onion.bbox,
                diameter_mm=round(diameter_mm, 1),
                is_calibrated=is_calibrated,
                visual_class=onion.visual_class,
                confidence=round(onion.confidence, 4),
                final_grade=final_grade,
                reason_codes=reason_codes,
                defects=defect_flags,
                polygon=onion.polygon,
            )
        )

    # 6. Compute Batch Summary
    batch_stats = grade_batch(results)

    # 7. Generate Evidence Reports (HTML, PDF, QR) if requested
    report_id = None
    if generate_artifacts:
        # Convert RGB for report generator
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        annot_rgb = cv2.cvtColor(detection.annotated_image, cv2.COLOR_BGR2RGB)
        report_id = generate_report(
            batch_id=batch_id,
            farmer_id=farmer_id,
            officer_id=officer_id,
            centre=centre,
            policy=policy_dict,
            results=results,
            original_image=orig_rgb,
            annotated_image=annot_rgb,
            gps=gps,
            output_dir=output_dir,
        )

    return {
        "status": "success" if detection.is_onion_frame else "no_onions_detected",
        "report_id": report_id,
        "is_onion_frame": detection.is_onion_frame,
        "summary": batch_stats,
        "calibration": cal_details,
        "onions": [r.model_dump() for r in results],
        "annotated_image": detection.annotated_image,
        "original_image": img_bgr,
        "filter_summary": detection.filter_summary,
    }
