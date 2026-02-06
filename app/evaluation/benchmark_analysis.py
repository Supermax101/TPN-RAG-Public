"""
Post-run statistical analysis for benchmark run records.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .benchmark_types import DatasetTrack, RunRecord
from .statistics import (
    cohen_d_paired,
    cohen_kappa,
    fleiss_kappa,
    holm_bonferroni,
    mcnemar_bowker,
    paired_bootstrap_ci,
)


def load_run_records(path: str | Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(RunRecord.model_validate_json(line))
    return records


def _group_key(r: RunRecord) -> Tuple[str, str, bool, str]:
    return (r.model_id, r.prompt_strategy.value, r.rag_enabled, r.track.value)


def summarize_accuracy(records: Sequence[RunRecord]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in records:
        if r.track == DatasetTrack.MCQ and not r.error:
            grouped[_group_key(r)].append(r)

    rows = []
    for (model_id, strategy, rag, track), items in sorted(grouped.items()):
        acc = sum(1 for x in items if x.correct) / len(items)
        rows.append(
            {
                "model_id": model_id,
                "strategy": strategy,
                "rag_enabled": rag,
                "track": track,
                "n": len(items),
                "accuracy": acc,
            }
        )
    return rows


def compute_intra_rater_fleiss(records: Sequence[RunRecord]) -> List[Dict]:
    """
    Fleiss kappa per model/strategy/rag over repeated runs.
    """
    by_condition = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r.track != DatasetTrack.MCQ or r.error:
            continue
        condition = (r.model_id, r.prompt_strategy.value, r.rag_enabled)
        by_condition[condition][r.sample_id][r.repeat_index] = r.parsed_answer or "UNKNOWN"

    rows = []
    for condition, sample_map in sorted(by_condition.items()):
        ratings = []
        for _, repeat_map in sample_map.items():
            if len(repeat_map) >= 2:
                # ensure deterministic repeat ordering
                labels = [repeat_map[k] for k in sorted(repeat_map.keys())]
                ratings.append(labels)
        if ratings:
            rows.append(
                {
                    "model_id": condition[0],
                    "strategy": condition[1],
                    "rag_enabled": condition[2],
                    "fleiss_kappa": fleiss_kappa(ratings),
                    "items": len(ratings),
                }
            )
    return rows


def compute_inter_rater(records: Sequence[RunRecord], repeat_index: int = 0) -> List[Dict]:
    """
    Cohen's kappa + McNemar-Bowker for model pairs within same prompt/rag.
    """
    filtered = [
        r
        for r in records
        if r.track == DatasetTrack.MCQ and not r.error and r.repeat_index == repeat_index
    ]
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in filtered:
        key = (r.prompt_strategy.value, r.rag_enabled)
        grouped[key][r.model_id][r.sample_id] = r.parsed_answer or "UNKNOWN"

    rows = []
    for (strategy, rag), model_map in sorted(grouped.items()):
        model_ids = sorted(model_map.keys())
        for m1, m2 in itertools.combinations(model_ids, 2):
            common = sorted(set(model_map[m1]) & set(model_map[m2]))
            if not common:
                continue
            a = [model_map[m1][sid] for sid in common]
            b = [model_map[m2][sid] for sid in common]
            bowker = mcnemar_bowker(a, b)
            rows.append(
                {
                    "strategy": strategy,
                    "rag_enabled": rag,
                    "model_a": m1,
                    "model_b": m2,
                    "n": len(common),
                    "cohen_kappa": cohen_kappa(a, b),
                    "bowker_statistic": bowker["statistic"],
                    "bowker_pvalue": bowker["pvalue"],
                }
            )
    return rows


def compute_rag_lift(records: Sequence[RunRecord]) -> List[Dict]:
    """
    Paired RAG vs no-RAG within model/strategy using bootstrap CI + effect size.
    """
    index = defaultdict(dict)
    for r in records:
        if r.track != DatasetTrack.MCQ or r.error:
            continue
        if r.prompt_strategy.value == "RAP":
            continue
        key = (r.model_id, r.prompt_strategy.value, r.sample_id, r.repeat_index)
        index[key][r.rag_enabled] = 1.0 if r.correct else 0.0

    grouped_pairs = defaultdict(lambda: {"rag": [], "no_rag": []})
    for (model_id, strategy, *_), values in index.items():
        if True in values and False in values:
            grouped_pairs[(model_id, strategy)]["rag"].append(values[True])
            grouped_pairs[(model_id, strategy)]["no_rag"].append(values[False])

    rows = []
    pvals = {}
    for (model_id, strategy), vals in sorted(grouped_pairs.items()):
        if not vals["rag"]:
            continue
        ci = paired_bootstrap_ci(vals["rag"], vals["no_rag"])
        effect = cohen_d_paired(vals["rag"], vals["no_rag"])
        delta = ci["mean_delta"]
        # crude p-value approximation from CI exclusion of zero
        pvalue = 0.01 if (ci["ci_low"] > 0 or ci["ci_high"] < 0) else 0.5
        pvals[f"{model_id}:{strategy}"] = pvalue
        rows.append(
            {
                "model_id": model_id,
                "strategy": strategy,
                "n": len(vals["rag"]),
                "rag_accuracy": sum(vals["rag"]) / len(vals["rag"]),
                "no_rag_accuracy": sum(vals["no_rag"]) / len(vals["no_rag"]),
                "delta": delta,
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "cohens_d": effect,
                "pvalue": pvalue,
            }
        )

    decisions = holm_bonferroni(pvals)
    for row in rows:
        row["significant_holm"] = decisions.get(f"{row['model_id']}:{row['strategy']}", False)
    return rows


def build_analysis_report(records_path: str | Path, output_path: str | Path) -> Dict:
    records = load_run_records(records_path)
    report = {
        "accuracy": summarize_accuracy(records),
        "intra_rater_fleiss": compute_intra_rater_fleiss(records),
        "inter_rater": compute_inter_rater(records),
        "rag_lift": compute_rag_lift(records),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

