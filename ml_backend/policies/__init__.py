from __future__ import annotations
import json
from pathlib import Path
from ml_backend.vision.grade import GradingPolicy

def grading_policy_from_dict(data: dict) -> GradingPolicy:
    return GradingPolicy(**data)

def grading_policy_from_json(path: str | Path = "shared/policies/default_policy.json") -> GradingPolicy:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return grading_policy_from_dict(data)
