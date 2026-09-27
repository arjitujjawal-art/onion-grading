from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class DefectFlags:
    damaged: bool = False
    rotten: bool = False
    sprouted: bool = False
    undersized: bool = False

@dataclass
class OnionResult:
    onion_id: str
    bbox: tuple[float, float, float, float]
    diameter_mm: float
    defects: DefectFlags
    confidence: float
    final_grade: Literal["GRADE_A", "GRADE_URS", "REJECTED", "MANUAL_REVIEW"]
    reason_codes: list[str] = field(default_factory=list)

@dataclass
class BatchReport:
    report_id: str
    batch_id: str
    farmer_id: str
    officer_id: str
    centre: str
    timestamp: str
    policy_version: str
    results: list[OnionResult]
    total: int
    grade_a_count: int
    grade_a_pct: float
    grade_urs_count: int
    grade_urs_pct: float
    rejected_count: int
    rejected_pct: float
    manual_review_count: int
    manual_review_pct: float
