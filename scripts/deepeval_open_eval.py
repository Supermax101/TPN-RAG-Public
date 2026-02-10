#!/usr/bin/env python3
"""
DeepEval scoring for open-ended benchmarks (QandA20 + Takeoff41-200).

This script is generation-model-agnostic: it consumes an existing benchmark
ledger (run_records_*.jsonl) and applies DeepEval judge metrics.

Supports multi-judge evaluation (OpenAI / Anthropic / Gemini) and reports basic
agreement statistics on the GEval correctness metric.

Typical usage (single judge):
  python scripts/deepeval_open_eval.py \
    --rubric qanda20 \
    --dataset eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl \
    --records eval/paper_runs/<RUN_SET_ID>/open/qanda20/<model>/<condition>/run_records_*.jsonl \
    --snapshots eval/cache/retrieval_snapshots_kbclean_qanda20.jsonl \
    --out-dir eval/results/deepeval/open/qanda20/<model>/<condition>

Tri-judge (secondary judges on a subset):
  python scripts/deepeval_open_eval.py \
    --rubric takeoff41 \
    --dataset eval/data/benchmark_2026-02-05/takeoff41_200_holdout.jsonl \
    --records <path/to/run_records_*.jsonl> \
    --snapshots eval/cache/retrieval_snapshots_kbclean_takeoff41_200.jsonl \
    --out-dir eval/results/deepeval/open/takeoff41_200 \
    --primary-judge openai:gpt-4.1-mini-2025-04-14 \
    --secondary-judge anthropic:claude-haiku-4-5-20251001 \
    --secondary-judge gemini:gemini-2.5-flash-lite \
    --secondary-subset-size 50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, get_args, get_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

_RUBRICS: Dict[str, str] = {
    "qanda20": (
        "Score the answer for clinical correctness compared to the expected output. "
        "Accept paraphrases and clinically equivalent wording. "
        "When the expected output includes numeric values (dose/rate/volume) the answer must match "
        "within normal clinical rounding tolerance and include correct units. "
        "Penalize contradictions, unsafe recommendations, or invented facts not supported by the question. "
        "Prefer concise, minimally sufficient answers."
    ),
    "takeoff41": (
        "Score the answer for correctness compared to the expected output. "
        "The answer must match the key facts from the expected output; accept paraphrases. "
        "Numeric thresholds/ranges/doses must match within normal clinical rounding tolerance and include units. "
        "Penalize invented numbers, incorrect product comparisons, or guidance that contradicts the expected output. "
        "Prefer concise, minimally sufficient answers."
    ),
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latest_records_file(root: Path) -> Path:
    files = sorted(root.glob("run_records_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No run_records_*.jsonl files in {root}")
    return files[-1]


# ---------------------------------------------------------------------------
# Stats helpers (no numpy/scipy dependency)
# ---------------------------------------------------------------------------

def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    if den <= 0:
        return None
    return float(num / den)


def _rankdata(values: Sequence[float]) -> List[float]:
    """
    Dense ranks with average tie handling. Ranks start at 1.0.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1])

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        v = indexed[i][1]
        while j < len(indexed) and indexed[j][1] == v:
            j += 1
        # Average rank for ties in [i, j)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 2:
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    p0 = agree / n
    pa = sum(1 for x in a if x) / n
    pb = sum(1 for y in b if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return None
    return float((p0 - pe) / (1 - pe))


# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeSpec:
    provider: str  # openai | anthropic | gemini
    model: str

    @property
    def judge_id(self) -> str:
        return f"{self.provider}:{self.model}"


def _parse_judge(spec: str) -> JudgeSpec:
    raw = (spec or "").strip()
    if ":" not in raw:
        raise ValueError(f"Invalid judge spec '{spec}'. Use provider:model (e.g. openai:gpt-5-mini-2025-08-07).")
    provider, model = raw.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider in {"gpt", "openai"}:
        provider = "openai"
    if provider in {"claude", "anthropic"}:
        provider = "anthropic"
    if provider in {"google", "gemini"}:
        provider = "gemini"
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError(f"Unsupported judge provider '{provider}' in '{spec}'.")
    if not model:
        raise ValueError(f"Invalid judge spec '{spec}': missing model name.")
    return JudgeSpec(provider=provider, model=model)


def _judge_llm(judge: JudgeSpec):
    # Late import for fast --dry-run.
    from deepeval.models import GeminiModel, GPTModel
    from deepeval.models.base_model import DeepEvalBaseLLM

    if judge.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai judge.")
        return GPTModel(model=judge.model, api_key=api_key, temperature=0.0)

    if judge.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic judge.")
        # DeepEval's native AnthropicModel relies on the judge emitting valid JSON
        # for schema-based metrics (Contextual Recall/Precision/Relevancy, etc.).
        # In practice, Claude models occasionally emit malformed or truncated JSON,
        # which hard-fails paper-grade tri-judge runs when --require-all-judges is set.
        #
        # This wrapper uses Anthropic tool-calling for any schema requests, so the
        # structured output is delivered as parsed JSON via tool_use.input and then
        # validated by Pydantic. This eliminates "invalid JSON" failures without
        # changing the judge model itself.

        class _AnthropicToolLLM(DeepEvalBaseLLM):
            def __init__(
                self,
                model: str,
                api_key: str,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ):
                self._api_key = api_key
                self.temperature = float(temperature)
                self._max_tokens = int(max_tokens)
                self._async_client = None
                super().__init__(model)

            def load_model(self, async_mode: bool = False):
                from anthropic import Anthropic, AsyncAnthropic

                if async_mode:
                    if self._async_client is None:
                        self._async_client = AsyncAnthropic(api_key=self._api_key)
                    return self._async_client
                return Anthropic(api_key=self._api_key)

            def get_model_name(self, *args, **kwargs) -> str:  # noqa: ANN001
                return self.name

            def _schema_to_input_schema(self, schema_cls) -> dict:  # noqa: ANN001
                # Keep this intentionally minimal: DeepEval schemas are simple
                # Pydantic models (nested objects, arrays, literals).
                from pydantic import BaseModel
                from typing import Any, Union, Literal

                def _type_to_schema(tp: Any) -> dict:
                    origin = get_origin(tp)
                    if origin in {list, List}:
                        args = get_args(tp)
                        item_tp = args[0] if args else Any
                        return {"type": "array", "items": _type_to_schema(item_tp)}
                    if origin is Union:
                        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
                        if len(args) == 1:
                            sch = _type_to_schema(args[0])
                            t = sch.get("type")
                            if isinstance(t, str):
                                sch["type"] = [t, "null"]
                            elif isinstance(t, list) and "null" not in t:
                                sch["type"] = list(t) + ["null"]
                            else:
                                sch["type"] = ["null"]
                            return sch
                        return {"anyOf": [_type_to_schema(a) for a in args]}
                    if origin is Literal:
                        lits = list(get_args(tp))
                        if all(isinstance(x, str) for x in lits):
                            t = "string"
                        elif all(isinstance(x, bool) for x in lits):
                            t = "boolean"
                        elif all(isinstance(x, int) for x in lits):
                            t = "integer"
                        elif all(isinstance(x, (int, float)) for x in lits):
                            t = "number"
                        else:
                            t = "string"
                        return {"type": t, "enum": lits}

                    if isinstance(tp, type) and issubclass(tp, BaseModel):
                        return _model_to_schema(tp)
                    if tp is str:
                        return {"type": "string"}
                    if tp is bool:
                        return {"type": "boolean"}
                    if tp is int:
                        return {"type": "integer"}
                    if tp is float:
                        return {"type": "number"}
                    return {}

                def _model_to_schema(model_cls: type[BaseModel]) -> dict:
                    props: dict = {}
                    required: list[str] = []
                    for name, field in (model_cls.model_fields or {}).items():
                        anno = getattr(field, "annotation", Any)
                        sch = _type_to_schema(anno)
                        # Pydantic allows defaults of None even when the annotation
                        # isn't Optional[...] (common in DeepEval schemas). Permit
                        # nulls for non-required fields with default None so the
                        # tool schema doesn't reject otherwise-valid outputs.
                        try:
                            default = getattr(field, "default", None)
                            is_req = getattr(field, "is_required", lambda: False)()
                            if (default is None) and (not is_req):
                                t = sch.get("type")
                                if isinstance(t, str):
                                    sch["type"] = [t, "null"]
                                elif isinstance(t, list) and "null" not in t:
                                    sch["type"] = list(t) + ["null"]
                        except Exception:
                            pass
                        props[name] = sch
                        if getattr(field, "is_required", lambda: False)():
                            required.append(name)
                    schema: dict = {"type": "object", "properties": props, "additionalProperties": False}
                    if required:
                        schema["required"] = required
                    return schema

                if not isinstance(schema_cls, type) or not issubclass(schema_cls, BaseModel):
                    # Shouldn't happen, but keep tool schema permissive.
                    return {"type": "object"}
                return _model_to_schema(schema_cls)

            def _extract_tool_input(self, message, tool_name: str) -> Optional[dict]:  # noqa: ANN001
                blocks = getattr(message, "content", None) or []
                for b in blocks:
                    # SDK block objects
                    if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == tool_name:
                        return getattr(b, "input", None)
                    # Dict blocks (defensive)
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == tool_name:
                        return b.get("input")
                return None

            def generate(self, prompt: str, schema=None):  # noqa: ANN001
                client = self.load_model(async_mode=False)
                if schema is None:
                    msg = client.messages.create(
                        model=self.name,
                        max_tokens=self._max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    blocks = getattr(msg, "content", None) or []
                    for b in blocks:
                        if getattr(b, "type", None) == "text":
                            return getattr(b, "text", "") or ""
                    return ""

                from pydantic import BaseModel

                tool_name = "structured_output"
                tool = {
                    "name": tool_name,
                    "description": "Return ONLY structured JSON that matches the provided schema.",
                    "input_schema": self._schema_to_input_schema(schema),
                }
                msg = client.messages.create(
                    model=self.name,
                    max_tokens=self._max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                )
                data = self._extract_tool_input(msg, tool_name) or {}
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    return schema.model_validate(data)
                return data

            async def a_generate(self, prompt: str, schema=None):  # noqa: ANN001
                client = self.load_model(async_mode=True)
                if schema is None:
                    msg = await client.messages.create(
                        model=self.name,
                        max_tokens=self._max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    blocks = getattr(msg, "content", None) or []
                    for b in blocks:
                        if getattr(b, "type", None) == "text":
                            return getattr(b, "text", "") or ""
                    return ""

                from pydantic import BaseModel

                tool_name = "structured_output"
                tool = {
                    "name": tool_name,
                    "description": "Return ONLY structured JSON that matches the provided schema.",
                    "input_schema": self._schema_to_input_schema(schema),
                }
                msg = await client.messages.create(
                    model=self.name,
                    max_tokens=self._max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                )
                data = self._extract_tool_input(msg, tool_name) or {}
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    return schema.model_validate(data)
                return data

        return _AnthropicToolLLM(model=judge.model, api_key=api_key, temperature=0.0, max_tokens=4096)

    if judge.provider == "gemini":
        # Some environments store Gemini keys as GOOGLE_API_KEY; accept either.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for gemini judge.")
        return GeminiModel(model=judge.model, api_key=api_key, temperature=0.0)

    raise ValueError(f"Unsupported judge provider: {judge.provider}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepEval scoring for open-ended benchmark runs")
    p.add_argument(
        "--rubric",
        type=str,
        default="qanda20",
        choices=sorted(_RUBRICS.keys()),
        help="Rubric preset: qanda20 or takeoff41",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Strict open-ended dataset JSONL path",
    )
    p.add_argument(
        "--records",
        type=str,
        default="",
        help="Path to run_records_*.jsonl (defaults to latest in --records-dir)",
    )
    p.add_argument(
        "--records-dir",
        type=str,
        default="eval/results/benchmark",
        help="Directory to search for latest run_records_*.jsonl if --records not provided",
    )
    p.add_argument(
        "--snapshots",
        type=str,
        default="",
        help="Optional retrieval snapshots JSONL (written by tpnctl precompute-retrieval). Required for RAG-only metrics.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="eval/results/deepeval/open",
        help="Output directory for DeepEval artifacts",
    )
    p.add_argument(
        "--primary-judge",
        type=str,
        default="openai:gpt-4.1-mini-2025-04-14",
        help="Primary judge in provider:model form (full coverage).",
    )
    p.add_argument(
        "--secondary-judge",
        type=str,
        action="append",
        default=[],
        help=(
            "Secondary judge(s) in provider:model form (optional). Can be passed multiple times. "
            "If omitted for rubric=qanda20, defaults to Claude Haiku 4.5 + Gemini 2.5 Flash Lite."
        ),
    )
    p.add_argument(
        "--secondary-subset-size",
        type=int,
        default=0,
        help="If >0, score only a random subset of this many cases for each secondary judge (primary judge scores all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic subset selection.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge calls (DeepEval AsyncConfig).",
    )
    p.add_argument(
        "--require-all-judges",
        action="store_true",
        help="Fail fast if any judge cannot be initialized or errors during scoring.",
    )
    p.add_argument(
        "--rag-only-when-context-used",
        action="store_true",
        help="Compute RAG grounding metrics only for cases where rag_context_used is true.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build test cases and write nothing; do not call judge models.",
    )
    return p.parse_args()


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    # Avoid find_dotenv() stack-walking issues; load explicitly from repo root.
    load_dotenv(PROJECT_ROOT / ".env")


def _metrics_map(test_result: dict) -> Dict[str, dict]:
    metrics = test_result.get("metrics_data") or test_result.get("metricsData") or []
    out: Dict[str, dict] = {}
    for m in metrics or []:
        name = m.get("name")
        if not name:
            continue
        out[str(name)] = {
            "score": m.get("score"),
            "success": m.get("success"),
            "threshold": m.get("threshold"),
            "reason": m.get("reason"),
        }
    return out


def _group_summary(rows: List[dict], meta_by_name: Dict[str, dict], kind: str) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        name = row.get("name") or row.get("testCase", {}).get("name")
        meta = meta_by_name.get(name or "", {})
        if not meta:
            continue
        key = (
            meta.get("model_id", ""),
            meta.get("strategy", ""),
            "rag" if meta.get("rag_enabled") else "no_rag",
        )
        grouped[key].append(row)

    out_rows: List[dict] = []
    for (model_id, strategy, rag_mode), items in sorted(grouped.items()):
        metrics = [_metrics_map(x) for x in items]
        metric_names = sorted({n for mm in metrics for n in mm.keys()})
        row: Dict[str, Any] = {
            "kind": kind,
            "model_id": model_id,
            "strategy": strategy,
            "rag_mode": rag_mode,
            "n": len(items),
        }
        for mn in metric_names:
            scores = [m[mn].get("score") for m in metrics if mn in m and m[mn].get("score") is not None]
            succ = [m[mn].get("success") for m in metrics if mn in m and m[mn].get("success") is not None]
            if scores:
                row[f"{mn}_mean"] = sum(float(s) for s in scores) / len(scores)
            if succ:
                row[f"{mn}_pass_rate"] = sum(1 for s in succ if bool(s)) / len(succ)
        out_rows.append(row)
    return out_rows


def _test_results(er) -> List[dict]:
    if er is None:
        return []
    raw = getattr(er, "model_dump", None)
    if callable(raw):
        data = er.model_dump()
    else:
        data = er.__dict__
    return list(data.get("test_results") or data.get("testResults") or [])


def main() -> int:
    args = parse_args()
    _maybe_load_dotenv()

    # Paper default: tri-judge on QandA20 unless the caller explicitly provides judges.
    if (not args.secondary_judge) and str(args.rubric).strip().lower() == "qanda20":
        args.secondary_judge = [
            "anthropic:claude-haiku-4-5-20251001",
            "gemini:gemini-2.5-flash-lite",
        ]

    dataset_path = (PROJECT_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    records_path = (
        (PROJECT_ROOT / args.records).resolve()
        if args.records
        else _latest_records_file((PROJECT_ROOT / args.records_dir).resolve())
    )
    snapshots_path = None
    if args.snapshots:
        snapshots_path = (PROJECT_ROOT / args.snapshots).resolve() if not Path(args.snapshots).is_absolute() else Path(args.snapshots)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Records not found: {records_path}")
    if snapshots_path and not snapshots_path.exists():
        raise FileNotFoundError(f"Snapshots not found: {snapshots_path}")

    dataset_rows = _load_jsonl(dataset_path)
    dataset_by_id = {r.get("sample_id"): r for r in dataset_rows if r.get("sample_id")}

    records_raw = _load_jsonl(records_path)
    open_records = [
        r
        for r in records_raw
        if str(r.get("track")) == "open_ended" and str(r.get("sample_id")) in dataset_by_id and not r.get("error")
    ]
    if not open_records:
        raise SystemExit(f"No open-ended records found for dataset {dataset_path.name} in {records_path.name}")

    snapshots_by_sample_id: Dict[str, dict] = {}
    if snapshots_path:
        for row in _load_jsonl(snapshots_path):
            sid = str(row.get("sample_id") or "")
            snap = row.get("snapshot")
            if sid and isinstance(snap, dict):
                snapshots_by_sample_id[sid] = snap

    if args.dry_run:
        rag = sum(1 for r in open_records if r.get("rag_enabled"))
        print(
            f"Dry run OK: dataset={len(dataset_rows)} open_records={len(open_records)} "
            f"rag_records={rag} snapshots={len(snapshots_by_sample_id)}"
        )
        return 0

    # Lazy imports to keep dry-run fast.
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build test cases (shared across judges) ---
    baseline_cases: List[LLMTestCase] = []
    rag_cases: List[LLMTestCase] = []
    meta_by_name: Dict[str, dict] = {}

    for r in open_records:
        sid = str(r["sample_id"])
        d = dataset_by_id[sid]

        rag_context_used = bool((r.get("metrics") or {}).get("rag_context_used"))
        tc = LLMTestCase(
            input=str(d.get("question") or ""),
            actual_output=str(r.get("response_text") or ""),
            expected_output=str(d.get("reference_answer") or ""),
            additional_metadata={
                "run_id": str(r.get("run_id") or ""),
                "sample_id": sid,
                "model_id": str(r.get("model_id") or ""),
                "strategy": str(r.get("prompt_strategy") or ""),
                "rag_enabled": bool(r.get("rag_enabled")),
                "rag_context_used": rag_context_used,
            },
        )
        # Keep stable for joins across judge runs.
        tc.name = f"{sid}:{r.get('run_id')}"
        meta_by_name[tc.name] = dict(tc.additional_metadata or {})

        if r.get("rag_enabled"):
            snap = snapshots_by_sample_id.get(sid)
            if snap:
                chunks = snap.get("chunks") or []
                retrieval_context = []
                for c in chunks:
                    src = str(c.get("source") or "unknown")
                    page = c.get("page")
                    content = str(c.get("content") or "")
                    header = f"{src}{f' (p.{page})' if page is not None else ''}"
                    retrieval_context.append(f"{header}\n{content}")
                tc.retrieval_context = retrieval_context
            rag_cases.append(tc)
        else:
            baseline_cases.append(tc)

    all_cases = baseline_cases + rag_cases
    rag_scored = [tc for tc in rag_cases if tc.retrieval_context]
    if args.rag_only_when_context_used:
        rag_scored = [tc for tc in rag_scored if bool((tc.additional_metadata or {}).get("rag_context_used"))]

    # Primary + optional secondary judges.
    primary = _parse_judge(args.primary_judge)
    secondary = [_parse_judge(s) for s in (args.secondary_judge or []) if str(s).strip()]

    # Secondary subset selection is done on the combined (baseline+rag) set so
    # agreement stats are computed on the exact same cases.
    rng = random.Random(int(args.seed))
    secondary_names: List[str] = [tc.name for tc in all_cases]
    if args.secondary_subset_size and args.secondary_subset_size > 0:
        k = min(int(args.secondary_subset_size), len(secondary_names))
        secondary_names = sorted(rng.sample(secondary_names, k=k))
    secondary_name_set = set(secondary_names)

    judges_to_run: List[Tuple[JudgeSpec, List[LLMTestCase], List[LLMTestCase]]] = []
    judges_to_run.append((primary, all_cases, rag_scored))
    for j in secondary:
        # Secondary judges can be run on a smaller subset for cost control.
        sub_all = [tc for tc in all_cases if tc.name in secondary_name_set]
        sub_rag = [tc for tc in rag_scored if tc.name in secondary_name_set]
        judges_to_run.append((j, sub_all, sub_rag))

    async_cfg = AsyncConfig(run_async=True, max_concurrent=args.max_concurrent)
    display_cfg = DisplayConfig(show_indicator=True, print_results=False, verbose_mode=False)

    CORRECTNESS_METRIC = "TPN_OpenCorrectness"
    rubric = _RUBRICS.get(args.rubric)
    if not rubric:
        raise SystemExit(f"Unknown rubric '{args.rubric}'.")

    per_judge_summaries: Dict[str, dict] = {}
    per_judge_scores: Dict[str, Dict[str, dict]] = {}  # judge_id -> tc.name -> {score, success}

    for judge, judge_cases, judge_rag_cases in judges_to_run:
        judge_id = judge.judge_id

        try:
            llm = _judge_llm(judge)
        except Exception as e:
            # Primary judge must work; secondary can be skipped (e.g. missing GEMINI_API_KEY).
            if args.require_all_judges or judge == primary:
                raise
            print(f"[WARN] Skipping judge {judge_id}: {e}", file=sys.stderr)
            continue

        # --- Metrics ---
        correctness = GEval(
            name=CORRECTNESS_METRIC,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            criteria=rubric,
            model=llm,
            threshold=0.8,
            async_mode=True,
            verbose_mode=False,
            _include_g_eval_suffix=False,
        )
        relevancy = AnswerRelevancyMetric(model=llm, threshold=0.8, async_mode=True)

        faithfulness = FaithfulnessMetric(model=llm, threshold=0.8, async_mode=True)
        c_precision = ContextualPrecisionMetric(model=llm, threshold=0.6, async_mode=True)
        c_recall = ContextualRecallMetric(model=llm, threshold=0.6, async_mode=True)
        c_relevancy = ContextualRelevancyMetric(model=llm, threshold=0.6, async_mode=True)

        try:
            result_base = evaluate(
                test_cases=judge_cases,
                metrics=[correctness, relevancy],
                async_config=async_cfg,
                display_config=display_cfg,
            )
        except Exception as e:
            if args.require_all_judges or judge == primary:
                raise
            print(
                f"[WARN] Base metrics failed for secondary judge {judge_id}: {e}",
                file=sys.stderr,
            )
            print(
                f"[INFO] Skipping secondary judge {judge_id} entirely.",
                file=sys.stderr,
            )
            continue

        result_rag = None
        if judge_rag_cases:
            try:
                result_rag = evaluate(
                    test_cases=judge_rag_cases,
                    metrics=[faithfulness, c_precision, c_recall, c_relevancy],
                    async_config=async_cfg,
                    display_config=display_cfg,
                )
            except Exception as e:
                if args.require_all_judges or judge == primary:
                    raise
                print(
                    f"[WARN] RAG grounding metrics failed for secondary judge {judge_id}: {e}",
                    file=sys.stderr,
                )
                print(
                    f"[INFO] Continuing with base metrics only for {judge_id}.",
                    file=sys.stderr,
                )

        base_rows = _test_results(result_base)
        rag_rows = _test_results(result_rag)

        # --- Serialize per-testcase results ---
        # Make output dirs portable across filesystems (macOS disallows ':' in filenames).
        judge_dirname = judge_id.replace("/", "_").replace(":", "__")
        judge_out_dir = out_dir / judge_dirname
        judge_out_dir.mkdir(parents=True, exist_ok=True)
        out_records_path = judge_out_dir / f"deepeval_records_{stamp}.jsonl"
        out_summary_path = judge_out_dir / f"deepeval_summary_{stamp}.json"
        out_summary_csv_path = judge_out_dir / f"deepeval_summary_{stamp}.csv"

        by_name = defaultdict(dict)
        for row in base_rows:
            name = row.get("name") or row.get("testCase", {}).get("name")
            if name:
                by_name[name]["base"] = row
        for row in rag_rows:
            name = row.get("name") or row.get("testCase", {}).get("name")
            if name:
                by_name[name]["rag"] = row

        with out_records_path.open("w", encoding="utf-8") as f:
            for name, payload in by_name.items():
                meta = meta_by_name.get(name, {})
                f.write(
                    json.dumps(
                        {
                            "name": name,
                            "meta": meta,
                            "base": payload.get("base"),
                            "rag": payload.get("rag"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        base_summary_rows = _group_summary(base_rows, meta_by_name, kind="base")
        rag_summary_rows = _group_summary(rag_rows, meta_by_name, kind="rag") if rag_rows else []

        summary = {
            "judge": {"provider": judge.provider, "model": judge.model, "judge_id": judge_id},
            "rubric": args.rubric,
            "dataset": str(dataset_path),
            "records": str(records_path),
            "snapshots": str(snapshots_path) if snapshots_path else None,
            "counts": {
                "open_records": len(open_records),
                "judge_cases": len(judge_cases),
                "judge_rag_cases": len(judge_rag_cases),
            },
            "summary_rows_base": base_summary_rows,
            "summary_rows_rag": rag_summary_rows,
        }
        out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        combined_rows = base_summary_rows + rag_summary_rows
        if combined_rows:
            fieldnames = list(dict.fromkeys(k for r in combined_rows for k in r.keys()))
            with out_summary_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in combined_rows:
                    w.writerow(r)

        per_judge_summaries[judge_id] = summary

        # Extract correctness scores for agreement stats.
        score_map: Dict[str, dict] = {}
        for row in base_rows:
            name = row.get("name") or row.get("testCase", {}).get("name")
            if not name:
                continue
            mm = _metrics_map(row)
            m = mm.get(CORRECTNESS_METRIC)
            if not m:
                continue
            score = m.get("score")
            success = m.get("success")
            if score is None:
                continue
            score_map[name] = {"score": float(score), "success": bool(success)}
        per_judge_scores[judge_id] = score_map

        print("DeepEval complete:")
        print(f"  Judge: {judge_id}")
        print(f"  Records: {out_records_path}")
        print(f"  Summary: {out_summary_path}")
        print(f"  Summary CSV: {out_summary_csv_path}")

    # --- Cross-judge agreement (on overlapping cases) ---
    judge_ids = sorted(per_judge_scores.keys())
    overlap: set[str] = set()
    if judge_ids:
        overlap = set(per_judge_scores[judge_ids[0]].keys())
        for jid in judge_ids[1:]:
            overlap &= set(per_judge_scores[jid].keys())

    agreement_rows: List[dict] = []
    disagreement_examples: List[dict] = []
    if len(judge_ids) >= 2 and overlap:
        overlap_sorted = sorted(overlap)
        # Pairwise stats.
        for i in range(len(judge_ids)):
            for j in range(i + 1, len(judge_ids)):
                a = judge_ids[i]
                b = judge_ids[j]
                xs = [per_judge_scores[a][n]["score"] for n in overlap_sorted]
                ys = [per_judge_scores[b][n]["score"] for n in overlap_sorted]
                xa = [per_judge_scores[a][n]["success"] for n in overlap_sorted]
                ya = [per_judge_scores[b][n]["success"] for n in overlap_sorted]
                agreement_rows.append(
                    {
                        "judge_a": a,
                        "judge_b": b,
                        "n": len(overlap_sorted),
                        "pearson": _pearson(xs, ys),
                        "spearman": _spearman(xs, ys),
                        "kappa_passfail": _cohen_kappa(xa, ya),
                    }
                )

        # Disagreement examples: max-min score gap across judges.
        gaps: List[Tuple[float, str, dict]] = []
        for name in overlap_sorted:
            scores = {jid: per_judge_scores[jid][name]["score"] for jid in judge_ids}
            successes = {jid: per_judge_scores[jid][name]["success"] for jid in judge_ids}
            gap = max(scores.values()) - min(scores.values())
            meta = meta_by_name.get(name, {})
            gaps.append(
                (
                    float(gap),
                    name,
                    {
                        "name": name,
                        "sample_id": meta.get("sample_id"),
                        "model_id": meta.get("model_id"),
                        "strategy": meta.get("strategy"),
                        "rag_enabled": meta.get("rag_enabled"),
                        "rag_context_used": meta.get("rag_context_used"),
                        "scores": scores,
                        "pass": successes,
                        "question_preview": str(dataset_by_id.get(meta.get("sample_id"), {}).get("question") or "")[:160],
                    },
                )
            )
        gaps.sort(key=lambda t: t[0], reverse=True)
        disagreement_examples = [item[2] for item in gaps[:5]]

    combined_summary = {
        "rubric": args.rubric,
        "dataset": str(dataset_path),
        "records": str(records_path),
        "snapshots": str(snapshots_path) if snapshots_path else None,
        "judges_ran": judge_ids,
        "agreement": agreement_rows,
        "top_disagreements": disagreement_examples,
        "per_judge": per_judge_summaries,
    }
    combined_path = out_dir / f"deepeval_open_combined_{stamp}.json"
    combined_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(f"\nCombined summary: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
