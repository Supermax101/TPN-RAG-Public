"""
Simple leakage checks for benchmark datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from .benchmark_types import DatasetSchema


def load_records(path: str | Path) -> List[DatasetSchema]:
    rows: List[DatasetSchema] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(DatasetSchema.model_validate(json.loads(line)))
    return rows


def check_data_leakage(records: List[DatasetSchema]) -> Dict[str, object]:
    """
    Check:
    1) exact question overlap between train/valid and holdout
    2) sample_id overlap between train/valid and holdout
    """
    by_split: Dict[str, List[DatasetSchema]] = {}
    for r in records:
        by_split.setdefault(r.split.value, []).append(r)

    holdout = by_split.get("holdout", [])
    train_like = by_split.get("train", []) + by_split.get("valid", []) + by_split.get("test", [])

    holdout_q: Set[str] = {r.question.strip().lower() for r in holdout}
    train_q: Set[str] = {r.question.strip().lower() for r in train_like}
    holdout_ids: Set[str] = {r.sample_id for r in holdout}
    train_ids: Set[str] = {r.sample_id for r in train_like}

    question_overlap = sorted(holdout_q & train_q)
    id_overlap = sorted(holdout_ids & train_ids)

    return {
        "total_records": len(records),
        "holdout_records": len(holdout),
        "train_like_records": len(train_like),
        "question_overlap_count": len(question_overlap),
        "sample_id_overlap_count": len(id_overlap),
        "question_overlap_examples": question_overlap[:20],
        "sample_id_overlap_examples": id_overlap[:20],
        "leakage_detected": bool(question_overlap or id_overlap),
    }

