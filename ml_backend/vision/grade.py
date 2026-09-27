"""
Configurable policy rule engine for onion grading.
Translates deep learning visual segmentation outputs and calibrated physical sizing
into government & mandi procurement classifications.
"""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Tuple
from collections import Counter
import numpy as np


class DefectFlags(BaseModel):
    damaged: bool = False
    rotten: bool = False
    sprouted: bool = False
    undersized: bool = False
    oversized: bool = False


class OnionGradingResult(BaseModel):
    onion_id: str
    bbox: Tuple[float, float, float, float]
    diameter_mm: float
    is_calibrated: bool = False
    visual_class: str = "grade_a"
    confidence: float = 0.0
    final_grade: Literal["GRADE_A", "GRADE_URS", "REJECTED", "MANUAL_REVIEW"] = "GRADE_A"
    reason_codes: List[str] = Field(default_factory=list)
    defects: DefectFlags = Field(default_factory=DefectFlags)
    polygon: Optional[List[List[float]]] = None

# Backwards compatibility alias
OnionResult = OnionGradingResult


class GradingPolicy(BaseModel):
    policy_id: str = "default_v1"
    version: str = "1.0.0"
    effective_date: str = "2026-09-27"
    urs_enabled: bool = True
    strict_calibration_required: bool = False
    min_confidence_threshold: float = 0.40
    grade_a: Dict[str, Any] = Field(
        default_factory=lambda: {
            "diameter_min_mm": 45.0,
            "diameter_max_mm": 65.0,
            "max_defects": {"damaged": False, "rotten": False, "sprouted": False},
        }
    )
    grade_urs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "diameter_min_mm": 35.0,
            "diameter_max_mm": 70.0,
            "max_defects": {"damaged": True, "rotten": False, "sprouted": True},
        }
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "GradingPolicy":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_policy(path: str | Path = "shared/policies/default_policy.json") -> GradingPolicy:
    if Path(path).is_file():
        return GradingPolicy.from_json(path)
    return GradingPolicy()


def grade_onion(
    diameter_mm: float,
    visual_class: str,
    confidence: float,
    calibration_ok: bool,
    policy: GradingPolicy,
) -> Tuple[Literal["GRADE_A", "GRADE_URS", "REJECTED", "MANUAL_REVIEW"], List[str], DefectFlags]:
    """
    Apply policy rules to evaluate a single detected onion.
    Returns: (final_grade, reason_codes, defect_flags)
    """
    reasons: List[str] = []
    defects = DefectFlags(
        damaged=(visual_class in ["reject", "grade_urs"]),
        rotten=(visual_class == "reject"),
        sprouted=False,
    )

    # 1. Low AI Confidence Check
    if confidence < policy.min_confidence_threshold:
        return "MANUAL_REVIEW", ["low_ai_confidence"], defects

    # 2. Strict Calibration Verification
    if policy.strict_calibration_required and not calibration_ok:
        return "MANUAL_REVIEW", ["calibration_reference_missing"], defects

    # If uncalibrated but policy allows estimation, note it
    if not calibration_ok:
        reasons.append("uncalibrated_size_estimate")

    # 3. Hard Visual Defect Reject
    if visual_class == "reject":
        reasons.append("visual_defect_reject")
        return "REJECTED", reasons, defects

    grade_a_cfg = policy.grade_a
    grade_urs_cfg = policy.grade_urs
    min_a = float(grade_a_cfg.get("diameter_min_mm", 45.0))
    max_a = float(grade_a_cfg.get("diameter_max_mm", 65.0))
    min_urs = float(grade_urs_cfg.get("diameter_min_mm", 35.0))
    max_urs = float(grade_urs_cfg.get("diameter_max_mm", 70.0))

    # 4. Grade URS Visual Class
    if visual_class == "grade_urs":
        reasons.append("slight_surface_imperfection")
        if not policy.urs_enabled:
            reasons.append("urs_scheme_disabled")
            return "REJECTED", reasons, defects

        if calibration_ok:
            if diameter_mm < min_urs:
                defects.undersized = True
                reasons.append(f"undersized_below_{min_urs:.0f}mm")
                return "REJECTED", reasons, defects
            elif diameter_mm > max_urs:
                defects.oversized = True
                reasons.append(f"oversized_above_{max_urs:.0f}mm")
                return "GRADE_URS", reasons, defects
            else:
                reasons.append(f"size_within_urs_{min_urs:.0f}-{max_urs:.0f}mm")
                return "GRADE_URS", reasons, defects
        else:
            return "GRADE_URS", reasons, defects

    # 5. Grade A Visual Class (Clean surface)
    if visual_class == "grade_a":
        reasons.append("visual_quality_prime")
        if calibration_ok:
            if min_a <= diameter_mm <= max_a:
                reasons.append(f"optimal_diameter_{min_a:.0f}-{max_a:.0f}mm")
                return "GRADE_A", reasons, defects
            elif min_urs <= diameter_mm < min_a:
                defects.undersized = True
                reasons.append(f"undersized_downgraded_to_urs_{diameter_mm:.1f}mm")
                return "GRADE_URS", reasons, defects
            elif max_a < diameter_mm <= max_urs:
                defects.oversized = True
                reasons.append(f"oversized_downgraded_to_urs_{diameter_mm:.1f}mm")
                return "GRADE_URS", reasons, defects
            else:
                reasons.append(f"diameter_out_of_bounds_{diameter_mm:.1f}mm")
                return "REJECTED", reasons, defects
        else:
            return "GRADE_A", reasons, defects

    return "MANUAL_REVIEW", ["unhandled_classification_state"], defects


def grade_batch(results: List[OnionGradingResult]) -> Dict[str, Any]:
    """
    Compute comprehensive procurement batch analytics and overall classification.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "counts": {"GRADE_A": 0, "GRADE_URS": 0, "REJECTED": 0, "MANUAL_REVIEW": 0},
            "grade_a_pct": 0.0,
            "grade_urs_pct": 0.0,
            "rejected_pct": 0.0,
            "manual_review_pct": 0.0,
            "avg_diameter_mm": 0.0,
            "overall_status": "NO_ONIONS",
            "acceptance_status": "EMPTY",
        }

    counts = Counter(r.final_grade for r in results)
    grade_a_cnt = counts.get("GRADE_A", 0)
    grade_urs_cnt = counts.get("GRADE_URS", 0)
    rej_cnt = counts.get("REJECTED", 0)
    manual_cnt = counts.get("MANUAL_REVIEW", 0)

    grade_a_pct = round((grade_a_cnt / total) * 100.0, 1)
    grade_urs_pct = round((grade_urs_cnt / total) * 100.0, 1)
    rej_pct = round((rej_cnt / total) * 100.0, 1)
    manual_pct = round((manual_cnt / total) * 100.0, 1)

    calibrated_diameters = [r.diameter_mm for r in results if r.is_calibrated and r.diameter_mm > 0]
    avg_diam = round(float(np.mean(calibrated_diameters)), 1) if calibrated_diameters else 0.0

    # Overall batch determination
    if rej_pct >= 25.0:
        overall_status = "REJECTED_BATCH"
        acceptance_status = "REJECT"
    elif manual_pct >= 30.0:
        overall_status = "MANUAL_INSPECTION_REQUIRED"
        acceptance_status = "REVIEW"
    elif grade_a_pct >= 70.0:
        overall_status = "PREMIUM_GRADE_A_BATCH"
        acceptance_status = "ACCEPT_GRADE_A"
    elif (grade_a_pct + grade_urs_pct) >= 70.0:
        overall_status = "COMMERCIAL_GRADE_URS_BATCH"
        acceptance_status = "ACCEPT_GRADE_URS"
    else:
        overall_status = "MIXED_QUALITY_BATCH"
        acceptance_status = "SORTING_REQUIRED"

    return {
        "total": total,
        "counts": {
            "GRADE_A": grade_a_cnt,
            "GRADE_URS": grade_urs_cnt,
            "REJECTED": rej_cnt,
            "MANUAL_REVIEW": manual_cnt,
        },
        "grade_a_pct": grade_a_pct,
        "grade_urs_pct": grade_urs_pct,
        "rejected_pct": rej_pct,
        "manual_review_pct": manual_pct,
        "avg_diameter_mm": avg_diam,
        "overall_status": overall_status,
        "acceptance_status": acceptance_status,
    }
