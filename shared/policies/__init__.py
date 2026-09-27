from __future__ import annotations
import json
from ml_backend.vision.grade import GradingPolicy

def grading_policy_from_dict(data: dict) -> GradingPolicy:
    return GradingPolicy(
        policy_id=data["policy_id"],
        version=data["version"],
        urs_enabled=data["urs_enabled"],
        grade_a=data["grade_a"],
        grade_urs=data["grade_urs"],
    )
