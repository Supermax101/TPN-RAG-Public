"""
Publishable benchmark runner:
- strict dataset schema validation
- fair shared retrieval snapshots for RAG conditions
- unified run ledger (RunRecord jsonl)
- per-model output directories with CSV accuracy summaries
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..logger import logger
from ..parsers.mcq_parser import MCQAnswer, answers_match, normalize_answer, parse_mcq_response
from .benchmark_types import (
    DatasetSchema,
    DatasetTrack,
    ExperimentConfig,
    ModelSpec,
    PromptStrategy,
    RetrievalSnapshot,
    RunRecord,
)
from ..prompting import get_system_prompt
from ..prompting import get_open_ended_system_prompt
from .metrics import AnswerMetrics
from .prompting import render_prompt
from .prompting import render_open_prompt
from .provider_adapter import PROVIDER_RATE_LIMITS, create_provider_adapter
from .retriever_adapter import RetrieverAdapter
from .calc_metrics import (
    analyze_reference_targets,
    evaluate_calc_metrics,
    evaluate_doc_citations,
    extract_final_answer_text,
)

# Number of independent passes for CoT-SC majority voting
_COT_SC_PASSES = 3

# Retry limit when parser returns UNKNOWN before structured output rescue
_UNKNOWN_RETRIES = 3

# JSON Schema for structured MCQ output (matches MCQAnswer Pydantic model)
_MCQ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string", "description": "Step-by-step clinical reasoning"},
        "answer": {"type": "string", "description": "Single letter answer: A, B, C, D, E, or F"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["thinking", "answer", "confidence"],
    "additionalProperties": False,
}


def _coerce_options(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        if "||" in raw:
            return [x.strip() for x in raw.split("||") if x.strip()]
        if "\n" in raw:
            lines = [x.strip() for x in raw.split("\n") if x.strip()]
            if len(lines) >= 2:
                return lines
        if "|" in raw:
            return [x.strip() for x in raw.split("|") if x.strip()]
    return None


def load_dataset(path: str | Path, track: DatasetTrack, require_holdout_only: bool = True) -> List[DatasetSchema]:
    """Load JSONL dataset and validate strict schema."""
    records: List[DatasetSchema] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)

            # Support both native benchmark schema and legacy {messages:[...]} format.
            if "messages" in raw:
                question = ""
                answer = ""
                for msg in raw.get("messages", []):
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        answer = msg.get("content", "")
                sample = {
                    "sample_id": str(raw.get("id", uuid.uuid4().hex[:12])),
                    "track": track.value,
                    "split": raw.get("split", "holdout"),
                    "question": question,
                    "reference_answer": answer if track == DatasetTrack.OPEN_ENDED else None,
                    "answer_key": raw.get("answer_key"),
                    "options": _coerce_options(raw.get("options")),
                    "domain": raw.get("domain"),
                    "proficiency": raw.get("proficiency"),
                    "source_doc": raw.get("source_doc"),
                    "page": raw.get("page"),
                    "metadata": raw,
                }
            else:
                sample = dict(raw)
                sample["track"] = track.value
                sample["options"] = _coerce_options(sample.get("options"))

            parsed = DatasetSchema.model_validate(sample)
            if require_holdout_only and parsed.split.value != "holdout":
                continue
            records.append(parsed)
    return records


class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(
        self,
        config: ExperimentConfig,
        retriever: Optional[RetrieverAdapter] = None,
        precomputed_snapshots: Optional[Dict[str, RetrievalSnapshot]] = None,
    ):
        self.config = config
        self.answer_metrics = AnswerMetrics()
        self._precomputed_snapshots = precomputed_snapshots or None

        # Optionally wrap retriever with agentic relevance judging
        if retriever and config.agentic_retrieval:
            from .agentic_retriever import AgenticRetrieverAdapter

            self.retriever = AgenticRetrieverAdapter(
                base_retriever=retriever,
                judge_provider=config.agentic_judge_provider,
                judge_model=config.agentic_judge_model,
            )
        else:
            self.retriever = retriever

        # Optionally init dynamic few-shot pool
        self._example_pool = None
        if config.dynamic_few_shot:
            from ..prompting.example_data import TPN_EXAMPLE_POOL
            from ..prompting.example_pool import FewShotPool

            self._example_pool = FewShotPool(TPN_EXAMPLE_POOL)

    def _mcq_max_tokens_for_strategy(self, strategy: PromptStrategy) -> int:
        if strategy == PromptStrategy.ZS:
            return int(self.config.mcq_max_tokens_zs)
        if strategy == PromptStrategy.FEW_SHOT:
            return int(self.config.mcq_max_tokens_few_shot)
        if strategy == PromptStrategy.COT:
            return int(self.config.mcq_max_tokens_cot)
        if strategy == PromptStrategy.COT_SC:
            return int(self.config.mcq_max_tokens_cot)
        # RAP is intentionally unsupported in current benchmark runs.
        return int(self.config.mcq_max_tokens_retry)

    def _should_use_rag_context(self, snapshot: Optional[RetrievalSnapshot]) -> tuple[bool, str, float]:
        """
        Decide whether to inject retrieved context into the prompt.

        Goal: prevent negative lift when retrieval is weak/out-of-coverage by
        falling back to the baseline (no-context) behavior.
        """
        if snapshot is None:
            return False, "no_snapshot", 0.0

        chunks = snapshot.chunks or []
        top_score = float(chunks[0].score) if chunks else 0.0

        context_text = (snapshot.context_text or "").strip()
        if not context_text:
            return False, "empty_context", top_score

        # Always-on safety: require at least some content.
        if snapshot.diagnostics.context_chars < self.config.rag_min_context_chars:
            return False, "context_too_short", top_score

        if not self.config.rag_gating_enabled:
            return True, "gating_disabled", top_score

        if snapshot.diagnostics.returned_count < self.config.rag_min_returned_chunks:
            return False, "too_few_chunks", top_score

        if top_score < self.config.rag_min_top_score:
            return False, "low_top_score", top_score

        return True, "pass", top_score

    def _condition_grid(self) -> List[Tuple[PromptStrategy, bool]]:
        conditions: List[Tuple[PromptStrategy, bool]] = []
        for strategy in self.config.prompt_strategies:
            # RAP is explicitly excluded from the current benchmark plan.
            if strategy == PromptStrategy.RAP:
                continue
            if self.config.include_no_rag and strategy != PromptStrategy.RAP:
                conditions.append((strategy, False))
            if self.config.include_rag:
                conditions.append((strategy, True))
        return conditions

    def _load_all_samples(self) -> List[DatasetSchema]:
        samples: List[DatasetSchema] = []
        if self.config.mcq_dataset_path:
            samples.extend(
                load_dataset(
                    self.config.mcq_dataset_path,
                    track=DatasetTrack.MCQ,
                    require_holdout_only=self.config.require_holdout_only,
                )
            )
        if self.config.open_dataset_path:
            samples.extend(
                load_dataset(
                    self.config.open_dataset_path,
                    track=DatasetTrack.OPEN_ENDED,
                    require_holdout_only=self.config.require_holdout_only,
                )
            )
        if not samples:
            raise ValueError("No evaluation samples loaded.")
        return samples

    async def _generate_single(
        self,
        adapter,
        prompt: str,
        model: ModelSpec,
        run_id: str,
        system_prompt: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = 800,
    ):
        """Make a single LLM call."""
        return await adapter.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_id=model.model_name,
            run_id=run_id,
            seed=seed,
        )

    async def _run_cot_sc(
        self,
        adapter,
        prompt: str,
        model: ModelSpec,
        run_id: str,
        system_prompt: str,
    ) -> tuple[str, int, float]:
        """
        Run CoT-SC: multiple independent passes at temp=0.7, majority-vote.

        Returns (response_text, tokens_used, latency_ms) where response_text
        is the text from the winning pass.
        """
        answers: List[str] = []
        texts: List[str] = []
        total_tokens = 0
        started = time.time()

        for i in range(_COT_SC_PASSES):
            result = await adapter.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=self._mcq_max_tokens_for_strategy(PromptStrategy.COT_SC),
                model_id=model.model_name,
                run_id=f"{run_id}_sc{i}",
            )
            texts.append(result.text)
            total_tokens += result.tokens_used
            parsed, _, _ = parse_mcq_response(result.text, is_multi_answer=False)
            answers.append(normalize_answer(parsed))

        latency_ms = (time.time() - started) * 1000

        # Majority vote
        vote_counts = Counter(answers)
        winner = vote_counts.most_common(1)[0][0]

        # Return text from the first pass that matched the winner
        best_text = texts[0]
        for ans, txt in zip(answers, texts):
            if ans == winner:
                best_text = txt
                break

        return best_text, total_tokens, latency_ms

    async def _run_one(
        self,
        model: ModelSpec,
        adapter,
        sample: DatasetSchema,
        strategy: PromptStrategy,
        rag_enabled: bool,
        repeat_index: int,
        retrieval_snapshot: Optional[RetrievalSnapshot],
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        # RAG context gating: when retrieval is weak, omit context and use the
        # baseline system prompt to avoid being misled by irrelevant text.
        rag_context_used = False
        rag_gate_reason = "no_rag"
        rag_top_score = 0.0

        context = retrieval_snapshot.context_text if retrieval_snapshot else None
        if rag_enabled:
            rag_context_used, rag_gate_reason, rag_top_score = self._should_use_rag_context(retrieval_snapshot)
            if not rag_context_used:
                context = None

        if sample.track == DatasetTrack.MCQ:
            system_prompt = get_system_prompt(use_rag=(rag_enabled and rag_context_used))
            prompt = render_prompt(
                strategy=strategy,
                question=sample.question,
                options=sample.options,
                context=context,
                example_pool=self._example_pool,
            )
        else:
            system_prompt = get_open_ended_system_prompt(use_rag=(rag_enabled and rag_context_used))
            prompt = render_open_prompt(
                strategy=strategy,
                question=sample.question,
                context=context,
            )

        started = time.time()
        error = None
        response_text = ""
        tokens_used = 0
        latency_ms = 0.0
        max_tokens = self._mcq_max_tokens_for_strategy(strategy) if sample.track == DatasetTrack.MCQ else 1000

        try:
            if strategy == PromptStrategy.COT_SC and sample.track == DatasetTrack.MCQ:
                # Real CoT-SC: 3 independent passes + majority vote
                # Intentionally omit seed for diversity at temp=0.7
                response_text, tokens_used, latency_ms = await self._run_cot_sc(
                    adapter, prompt, model, run_id, system_prompt,
                )
            else:
                result = await self._generate_single(
                    adapter, prompt, model, run_id, system_prompt, seed=self.config.seed,
                    max_tokens=max_tokens,
                )
                response_text = result.text
                tokens_used = result.tokens_used
                latency_ms = result.latency_ms
        except Exception as e:
            error = str(e)
            latency_ms = (time.time() - started) * 1000

        parsed_answer = None
        correct = None
        metrics: Dict[str, object] = {
            "rag_context_used": rag_context_used,
            "rag_gate_reason": rag_gate_reason,
            "rag_top_score": rag_top_score,
            "rag_context_chars": retrieval_snapshot.diagnostics.context_chars if retrieval_snapshot else 0,
            "rag_returned_chunks": retrieval_snapshot.diagnostics.returned_count if retrieval_snapshot else 0,
        }
        structured_output_used = False

        if not error:
            if sample.track == DatasetTrack.MCQ:
                # Parse answer: regex first, retry on UNKNOWN, structured rescue last
                parsed_answer, _, _ = parse_mcq_response(response_text, is_multi_answer=True)

                # Retry up to _UNKNOWN_RETRIES times when parser returns UNKNOWN
                if parsed_answer == "UNKNOWN":
                    for retry_i in range(_UNKNOWN_RETRIES):
                        try:
                            retry_result = await self._generate_single(
                                adapter, prompt, model, f"{run_id}_retry{retry_i}",
                                system_prompt,
                                seed=None,  # vary output
                                max_tokens=int(self.config.mcq_max_tokens_retry),
                            )
                            retry_answer, _, _ = parse_mcq_response(
                                retry_result.text, is_multi_answer=True,
                            )
                            if retry_answer != "UNKNOWN":
                                response_text = retry_result.text
                                parsed_answer = retry_answer
                                tokens_used += retry_result.tokens_used
                                break
                        except Exception:
                            pass  # continue retrying

                if parsed_answer == "UNKNOWN":
                    # All retries failed — structured output rescue
                    try:
                        structured = await adapter.generate_structured(
                            prompt=prompt,
                            schema=_MCQ_SCHEMA,
                            system=system_prompt,
                            temperature=0.0,
                            max_tokens=int(self.config.mcq_max_tokens_retry),
                            model_id=model.model_name,
                        )
                        parsed_answer = normalize_answer(structured.get("answer", ""))
                        structured_output_used = True
                    except Exception:
                        pass  # Keep UNKNOWN

                expected = normalize_answer(sample.answer_key or "")
                actual = normalize_answer(parsed_answer)
                exact, partial = answers_match(actual, expected)
                correct = bool(exact)
                metrics.update(
                    {
                        "exact_match": float(exact),
                        "partial_match": float(partial),
                    }
                )
            else:
                # Enforce a strict output contract for open-ended benchmarks:
                # - final answer only (no reasoning/work)
                # - no citations/sources
                #
                # This keeps DeepEval/GEval scoring clean and makes downstream
                # benchmarking artifacts easier to audit. If the model violates
                # the contract, do a single retry with an explicit correction.
                from .format_metrics import validate_open_final_answer

                fmt = validate_open_final_answer(response_text)
                metrics.update(
                    {
                        "format_ok": bool(fmt.ok),
                        "format_retry_used": False,
                        "format_violation_reason": fmt.reason,
                    }
                )

                if not fmt.ok:
                    prompt_retry = (
                        prompt.rstrip()
                        + "\n\nIMPORTANT: Your previous response violated the STRICT output rules. "
                        + "Output ONLY the final answer starting with exactly 'Final answer:' and nothing else. "
                        + "Do NOT include citations, sources, brackets, analysis, reasoning, work, or extra headers.\n"
                        + "Final answer:"
                    )
                    try:
                        retry = await self._generate_single(
                            adapter=adapter,
                            prompt=prompt_retry,
                            model=model,
                            run_id=f"{run_id}_format_retry",
                            system_prompt=system_prompt,
                            seed=None,  # vary output; some models ignore seed at temp=0
                            max_tokens=max_tokens,
                        )
                        response_text = retry.text
                        tokens_used += int(retry.tokens_used or 0)
                        latency_ms += float(retry.latency_ms or 0.0)
                        prompt = prompt_retry
                        metrics["format_retry_used"] = True

                        fmt2 = validate_open_final_answer(response_text)
                        metrics["format_ok"] = bool(fmt2.ok)
                        if not fmt2.ok:
                            metrics["format_violation_reason_after_retry"] = fmt2.reason
                    except Exception as e:
                        metrics["format_retry_used"] = True
                        metrics["format_retry_error"] = str(e)

                eval_result = self.answer_metrics.evaluate_single(
                    question=sample.question,
                    generated=response_text,
                    ground_truth=sample.reference_answer or "",
                    ground_truth_source=sample.source_doc,
                    ground_truth_page=sample.page,
                )
                calc_full = evaluate_calc_metrics(
                    expected_answer=sample.reference_answer or "",
                    output_answer=response_text,
                )
                final_text = extract_final_answer_text(response_text)
                calc_final = evaluate_calc_metrics(
                    expected_answer=sample.reference_answer or "",
                    output_answer=final_text,
                )
                ref = analyze_reference_targets(sample.reference_answer or "")
                retrieved_sources = (
                    [c.source for c in retrieval_snapshot.chunks] if retrieval_snapshot else []
                )
                citations = evaluate_doc_citations(
                    output_answer=response_text,
                    gold_source_doc=sample.source_doc,
                    retrieved_sources=retrieved_sources,
                )
                metrics.update(
                    {
                        "f1": eval_result.f1_score,
                        "exact_match": eval_result.exact_match,
                        "key_phrase_overlap": eval_result.key_phrase_overlap,
                        "citation_match": float(eval_result.citation_match),
                        # Deterministic calc metrics (full output).
                        "quantity_recall": calc_full.quantity_recall,
                        "quantity_precision": calc_full.quantity_precision,
                        "quantity_f1": calc_full.quantity_f1,
                        "key_recall": calc_full.key_recall,
                        "key_precision": calc_full.key_precision,
                        "key_f1": calc_full.key_f1,
                        "unit_mismatch_count": calc_full.unit_mismatch_count,
                        "expected_quantity_count": calc_full.expected_quantity_count,
                        "output_quantity_count": calc_full.output_quantity_count,
                        "matched_quantity_count": calc_full.matched_quantity_count,
                        "expected_key_count": calc_full.expected_key_count,
                        "output_key_count": calc_full.output_key_count,
                        "matched_key_count": calc_full.matched_key_count,
                        "rel_error_mean": calc_full.rel_error_mean,
                        "rel_error_p50": calc_full.rel_error_p50,
                        "rel_error_p95": calc_full.rel_error_p95,
                        # Deterministic calc metrics (Final answer block only).
                        "final_quantity_recall": calc_final.quantity_recall,
                        "final_quantity_precision": calc_final.quantity_precision,
                        "final_quantity_f1": calc_final.quantity_f1,
                        "final_key_recall": calc_final.key_recall,
                        "final_key_precision": calc_final.key_precision,
                        "final_key_f1": calc_final.key_f1,
                        "final_unit_mismatch_count": calc_final.unit_mismatch_count,
                        "final_output_quantity_count": calc_final.output_quantity_count,
                        "final_output_key_count": calc_final.output_key_count,
                        "final_rel_error_mean": calc_final.rel_error_mean,
                        "final_rel_error_p50": calc_final.rel_error_p50,
                        "final_rel_error_p95": calc_final.rel_error_p95,
                        # Reference "single-target" indicators (silver-label robustness reporting).
                        "ref_multi_value_key_count": ref.multi_value_key_count,
                        "ref_is_single_target": float(ref.is_single_target),
                        "doc_citation_present": float(citations.citation_present),
                        "doc_cited_doc_count": citations.cited_doc_count,
                        "doc_cites_gold_source": float(citations.cites_gold_source_doc)
                        if citations.cites_gold_source_doc is not None
                        else None,
                        "doc_cited_in_retrieved_context": float(citations.cited_doc_in_retrieved_context),
                    }
                )

        return RunRecord(
            run_id=run_id,
            sample_id=sample.sample_id,
            track=sample.track,
            model_id=model.model_id,
            model_name=model.model_name,
            provider=model.provider,
            model_tier=model.tier,
            prompt_strategy=strategy,
            rag_enabled=rag_enabled,
            repeat_index=repeat_index,
            question=sample.question,
            prompt=prompt,
            response_text=response_text,
            parsed_answer=parsed_answer,
            correct=correct,
            retrieval_context_hash=retrieval_snapshot.context_hash if retrieval_snapshot else None,
            retrieval_snapshot_id=retrieval_snapshot.run_id if retrieval_snapshot else None,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metrics=metrics,
            structured_output_used=structured_output_used,
            error=error,
        )

    async def _gated_run(
        self,
        semaphore: asyncio.Semaphore,
        delay: float,
        model: ModelSpec,
        adapter,
        sample: DatasetSchema,
        strategy: PromptStrategy,
        rag_enabled: bool,
        repeat_index: int,
        retrieval_snapshot: Optional[RetrievalSnapshot],
        progress: Optional[Dict[str, int]] = None,
    ) -> RunRecord:
        """Run a single benchmark call gated by a per-provider semaphore + delay."""
        async with semaphore:
            if delay > 0:
                await asyncio.sleep(delay)
            record = await self._run_one(
                model=model,
                adapter=adapter,
                sample=sample,
                strategy=strategy,
                rag_enabled=rag_enabled,
                repeat_index=repeat_index,
                retrieval_snapshot=retrieval_snapshot,
            )
            if progress is not None:
                progress["done"] += 1
                done = progress["done"]
                total = progress["total"]
                if done % 10 == 0 or done == total:
                    pct = done / total * 100
                    logger.info("Progress: %d/%d (%.0f%%)", done, total, pct)
            return record

    async def run(self) -> Dict[str, object]:
        """
        Execute full benchmark matrix and return summary + record paths.

        ALL calls across all samples are parallelized using asyncio.gather
        with per-provider semaphores and inter-request delays.
        """
        samples = self._load_all_samples()
        models = [m for m in self.config.models if m.enabled]
        conditions = self._condition_grid()

        total_calls = len(samples) * len(conditions) * self.config.repeats * len(models)
        logger.info(
            "Benchmark start: %d samples × %d conditions × %d repeats × %d models = %d total calls",
            len(samples),
            len(conditions),
            self.config.repeats,
            len(models),
            total_calls,
        )

        adapters = {
            m.model_id: create_provider_adapter(m.provider, m.model_name, m.api_key_env)
            for m in models
        }

        # Per-provider semaphores + delays from rate limit config
        provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        provider_delays: Dict[str, float] = {}
        for m in models:
            p = m.provider.lower()
            if p not in provider_semaphores:
                limits = PROVIDER_RATE_LIMITS.get(p, {"max_concurrent": self.config.max_concurrent, "delay": 0.0})
                provider_semaphores[p] = asyncio.Semaphore(limits["max_concurrent"])
                provider_delays[p] = limits["delay"]

        records: List[RunRecord] = []
        retrieval_cache_shared: Dict[str, RetrievalSnapshot] = {}
        retrieval_cache_per_strategy: Dict[Tuple[str, str], RetrievalSnapshot] = {}
        progress = {"done": 0, "total": total_calls}

        # If snapshots were precomputed offline, load them and skip retrieval.
        if self._precomputed_snapshots is not None and self.config.include_rag:
            retrieval_cache_shared = dict(self._precomputed_snapshots)

        # Pre-compute retrieval snapshots (sequential — disk I/O)
        # When fair_shared_context is enabled, reuse the same snapshot across all
        # prompt strategies since retrieval is independent of prompting strategy.
        if self.retriever and self.config.include_rag:
            if self.config.fair_shared_context:
                for sample in samples:
                    if sample.sample_id in retrieval_cache_shared:
                        continue
                    retrieval_cache_shared[sample.sample_id] = self.retriever.retrieve(
                        query=sample.question,
                        query_id=sample.sample_id,
                        run_id=uuid.uuid4().hex,
                    )
            else:
                for sample in samples:
                    for strategy, rag_enabled in conditions:
                        if not rag_enabled:
                            continue
                        key = (sample.sample_id, strategy.value)
                        if key in retrieval_cache_per_strategy:
                            continue
                        retrieval_cache_per_strategy[key] = self.retriever.retrieve(
                            query=sample.question,
                            query_id=sample.sample_id,
                            run_id=uuid.uuid4().hex,
                        )

        # Build ALL tasks upfront, then gather
        all_tasks = []
        for sample in samples:
            for strategy, rag_enabled in conditions:
                snapshot = None
                # Snapshot selection must not depend on having a live retriever.
                # When --retrieval-snapshots-in is used, self.retriever is None
                # but retrieval_cache_* is populated from the precomputed file.
                if rag_enabled:
                    if self.config.fair_shared_context:
                        snapshot = retrieval_cache_shared.get(sample.sample_id)
                    else:
                        key = (sample.sample_id, strategy.value)
                        snapshot = retrieval_cache_per_strategy.get(key)

                for repeat_idx in range(self.config.repeats):
                    for model in models:
                        p = model.provider.lower()
                        all_tasks.append(
                            self._gated_run(
                                semaphore=provider_semaphores[p],
                                delay=provider_delays[p],
                                model=model,
                                adapter=adapters[model.model_id],
                                sample=sample,
                                strategy=strategy,
                                rag_enabled=rag_enabled,
                                repeat_index=repeat_idx,
                                retrieval_snapshot=snapshot,
                                progress=progress,
                            )
                        )

        logger.info("Dispatching %d async calls with per-provider rate limits...", len(all_tasks))
        records = list(await asyncio.gather(*all_tasks))

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        # --- Global combined files (backward compat) ---
        ledger_path = output_dir / f"run_records_{stamp}.jsonl"
        with ledger_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

        summary = self._summarize(records)
        summary_path = output_dir / f"summary_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # --- Per-model output directories ---
        model_records: Dict[str, List[RunRecord]] = defaultdict(list)
        for r in records:
            model_records[r.model_id].append(r)

        model_dirs = []
        for model_id, m_records in model_records.items():
            model_dir = output_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Per-model JSONL (full responses)
            m_ledger = model_dir / f"run_records_{stamp}.jsonl"
            with m_ledger.open("w", encoding="utf-8") as f:
                for r in m_records:
                    f.write(r.model_dump_json() + "\n")

            # Per-model summary JSON
            m_summary = self._summarize(m_records)
            m_summary_path = model_dir / f"summary_{stamp}.json"
            m_summary_path.write_text(json.dumps(m_summary, indent=2), encoding="utf-8")

            # Per-model accuracy CSV
            m_csv_path = model_dir / f"accuracy_{stamp}.csv"
            self._write_accuracy_csv(m_csv_path, m_summary)

            model_dirs.append(str(model_dir))
            logger.info("  Model %s → %s", model_id, model_dir)

        # --- Global accuracy CSV (all models combined) ---
        global_csv_path = output_dir / f"accuracy_{stamp}.csv"
        self._write_accuracy_csv(global_csv_path, summary)

        logger.info(
            "Benchmark complete. Records: %s, Summary: %s, CSV: %s",
            ledger_path, summary_path, global_csv_path,
        )
        return {
            "records_path": str(ledger_path),
            "summary_path": str(summary_path),
            "csv_path": str(global_csv_path),
            "model_dirs": model_dirs,
            "summary": summary,
        }

    def _summarize(self, records: Sequence[RunRecord]) -> Dict[str, object]:
        grouped = defaultdict(list)
        for r in records:
            key = (r.model_id, r.prompt_strategy.value, "rag" if r.rag_enabled else "no_rag", r.track.value)
            grouped[key].append(r)

        table = []
        for (model_id, strategy, rag_mode, track), rows in sorted(grouped.items()):
            valid = [r for r in rows if not r.error]
            if not valid:
                continue
            row = {
                "model_id": model_id,
                "strategy": strategy,
                "rag_mode": rag_mode,
                "track": track,
                "n": len(valid),
                "latency_ms_mean": sum(r.latency_ms for r in valid) / len(valid),
                "error_rate": (len(rows) - len(valid)) / len(rows),
            }
            row["rag_context_used_rate"] = (
                sum(float(r.metrics.get("rag_context_used", 0.0)) for r in valid) / len(valid)
            )
            if rag_mode == "rag":
                used = [r for r in valid if bool(r.metrics.get("rag_context_used"))]
                not_used = [r for r in valid if not bool(r.metrics.get("rag_context_used"))]
                row["n_ctx_used"] = len(used)
                row["n_ctx_not_used"] = len(not_used)
            if track == DatasetTrack.MCQ.value:
                row["accuracy"] = sum(1 for r in valid if r.correct) / len(valid)
                row["partial_rate"] = (
                    sum(float(r.metrics.get("partial_match", 0.0)) for r in valid) / len(valid)
                )
                if rag_mode == "rag":
                    used = [r for r in valid if bool(r.metrics.get("rag_context_used"))]
                    not_used = [r for r in valid if not bool(r.metrics.get("rag_context_used"))]
                    if used:
                        row["accuracy_ctx_used"] = sum(1 for r in used if r.correct) / len(used)
                    if not_used:
                        row["accuracy_ctx_not_used"] = sum(1 for r in not_used if r.correct) / len(not_used)
            else:
                row["f1_mean"] = sum(float(r.metrics.get("f1", 0.0)) for r in valid) / len(valid)
                row["citation_match_rate"] = (
                    sum(float(r.metrics.get("citation_match", 0.0)) for r in valid) / len(valid)
                )
                row["quantity_f1_mean"] = (
                    sum(float(r.metrics.get("quantity_f1", 0.0)) for r in valid) / len(valid)
                )
                row["quantity_recall_mean"] = (
                    sum(float(r.metrics.get("quantity_recall", 0.0)) for r in valid) / len(valid)
                )
                row["quantity_precision_mean"] = (
                    sum(float(r.metrics.get("quantity_precision", 0.0)) for r in valid) / len(valid)
                )
                row["key_f1_mean"] = (
                    sum(float(r.metrics.get("key_f1", 0.0)) for r in valid) / len(valid)
                )
                row["key_recall_mean"] = (
                    sum(float(r.metrics.get("key_recall", 0.0)) for r in valid) / len(valid)
                )
                row["key_precision_mean"] = (
                    sum(float(r.metrics.get("key_precision", 0.0)) for r in valid) / len(valid)
                )
                row["final_quantity_f1_mean"] = (
                    sum(float(r.metrics.get("final_quantity_f1", 0.0)) for r in valid) / len(valid)
                )
                row["final_key_f1_mean"] = (
                    sum(float(r.metrics.get("final_key_f1", 0.0)) for r in valid) / len(valid)
                )
                row["ref_single_target_rate"] = (
                    sum(float(r.metrics.get("ref_is_single_target", 0.0)) for r in valid) / len(valid)
                )
                single = [r for r in valid if float(r.metrics.get("ref_is_single_target", 0.0) or 0.0) >= 0.5]
                if single:
                    row["final_key_f1_mean_single_target"] = (
                        sum(float(r.metrics.get("final_key_f1", 0.0)) for r in single) / len(single)
                    )
                    row["final_quantity_f1_mean_single_target"] = (
                        sum(float(r.metrics.get("final_quantity_f1", 0.0)) for r in single) / len(single)
                    )
                if rag_mode == "rag":
                    used = [r for r in valid if bool(r.metrics.get("rag_context_used"))]
                    not_used = [r for r in valid if not bool(r.metrics.get("rag_context_used"))]
                    if used:
                        row["final_key_f1_mean_ctx_used"] = (
                            sum(float(r.metrics.get("final_key_f1", 0.0)) for r in used) / len(used)
                        )
                    if not_used:
                        row["final_key_f1_mean_ctx_not_used"] = (
                            sum(float(r.metrics.get("final_key_f1", 0.0)) for r in not_used) / len(not_used)
                        )
                row["doc_citation_present_rate"] = (
                    sum(float(r.metrics.get("doc_citation_present", 0.0)) for r in valid) / len(valid)
                )
                gold_vals = [
                    float(v)
                    for r in valid
                    for v in [r.metrics.get("doc_cites_gold_source")]
                    if v is not None
                ]
                if gold_vals:
                    row["doc_cites_gold_source_rate"] = sum(gold_vals) / len(gold_vals)
                row["doc_cited_in_retrieved_context_rate"] = (
                    sum(float(r.metrics.get("doc_cited_in_retrieved_context", 0.0)) for r in valid)
                    / len(valid)
                )
            table.append(row)

        return {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "repeats": self.config.repeats,
            "rows": table,
        }

    @staticmethod
    def _write_accuracy_csv(path: Path, summary: Dict[str, object]) -> None:
        """Write summary rows to a CSV file for easy analysis."""
        rows = summary.get("rows", [])
        if not rows:
            return
        # Collect all column names across all rows
        fieldnames = list(dict.fromkeys(
            col for row in rows for col in row.keys()
        ))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


async def run_benchmark(config: ExperimentConfig, retriever: Optional[RetrieverAdapter] = None):
    """Convenience helper for script entrypoints."""
    runner = BenchmarkRunner(config=config, retriever=retriever)
    return await runner.run()
