"""
Publishable benchmark runner:
- strict dataset schema validation
- fair shared retrieval snapshots for RAG conditions
- unified run ledger (RunRecord jsonl)
"""

from __future__ import annotations

import asyncio
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
from .metrics import AnswerMetrics
from .prompting import render_prompt
from .provider_adapter import create_provider_adapter
from .retriever_adapter import RetrieverAdapter

# Board-certified TPN specialist system prompt (from app/chains/tpn_prompts.py)
_TPN_SYSTEM_PROMPT = (
    "You are a board-certified TPN (Total Parenteral Nutrition) Clinical Specialist "
    "with expertise in neonatal and pediatric nutrition support. You are taking the "
    "ASPEN Nutrition Support Certification exam.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. ALWAYS prioritize the Clinical Knowledge Base when provided.\n"
    "2. Your answers MUST be grounded in the provided context.\n"
    "3. If the context contradicts your training, TRUST THE CONTEXT.\n"
    "4. For 'FALSE' or 'LEAST likely' questions, identify the INCORRECT statement.\n"
    "5. Follow the output format precisely."
)

# Number of independent passes for CoT-SC majority voting
_COT_SC_PASSES = 3

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
    ):
        self.config = config
        self.answer_metrics = AnswerMetrics()

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

    def _condition_grid(self) -> List[Tuple[PromptStrategy, bool]]:
        conditions: List[Tuple[PromptStrategy, bool]] = []
        for strategy in self.config.prompt_strategies:
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
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Make a single LLM call."""
        return await adapter.generate(
            prompt=prompt,
            system=_TPN_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=1000,
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
                system=_TPN_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1000,
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
        context = retrieval_snapshot.context_text if retrieval_snapshot else None
        prompt = render_prompt(
            strategy=strategy,
            question=sample.question,
            options=sample.options,
            context=context,
            example_pool=self._example_pool,
        )

        started = time.time()
        error = None
        response_text = ""
        tokens_used = 0
        latency_ms = 0.0

        try:
            if strategy == PromptStrategy.COT_SC and sample.track == DatasetTrack.MCQ:
                # Real CoT-SC: 3 independent passes + majority vote
                # Intentionally omit seed for diversity at temp=0.7
                response_text, tokens_used, latency_ms = await self._run_cot_sc(
                    adapter, prompt, model, run_id,
                )
            else:
                result = await self._generate_single(
                    adapter, prompt, model, run_id, seed=self.config.seed,
                )
                response_text = result.text
                tokens_used = result.tokens_used
                latency_ms = result.latency_ms
        except Exception as e:
            error = str(e)
            latency_ms = (time.time() - started) * 1000

        parsed_answer = None
        correct = None
        metrics: Dict[str, float] = {}
        structured_output_used = False

        if not error:
            if sample.track == DatasetTrack.MCQ:
                # Parse answer: regex first, structured output only if regex fails
                parsed_answer, _, _ = parse_mcq_response(response_text, is_multi_answer=True)

                if parsed_answer == "PARSE_ERROR":
                    # Regex failed — try structured output as rescue
                    try:
                        structured = await adapter.generate_structured(
                            prompt=prompt,
                            schema=_MCQ_SCHEMA,
                            system=_TPN_SYSTEM_PROMPT,
                            temperature=0.0,
                            max_tokens=1000,
                            model_id=model.model_name,
                        )
                        parsed_answer = normalize_answer(structured.get("answer", ""))
                        structured_output_used = True
                    except Exception:
                        pass  # Keep PARSE_ERROR

                expected = normalize_answer(sample.answer_key or "")
                actual = normalize_answer(parsed_answer)
                exact, partial = answers_match(actual, expected)
                correct = bool(exact)
                metrics = {
                    "exact_match": float(exact),
                    "partial_match": float(partial),
                }
            else:
                eval_result = self.answer_metrics.evaluate_single(
                    question=sample.question,
                    generated=response_text,
                    ground_truth=sample.reference_answer or "",
                    ground_truth_source=sample.source_doc,
                    ground_truth_page=sample.page,
                )
                metrics = {
                    "f1": eval_result.f1_score,
                    "exact_match": eval_result.exact_match,
                    "key_phrase_overlap": eval_result.key_phrase_overlap,
                    "citation_match": float(eval_result.citation_match),
                }

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
        model: ModelSpec,
        adapter,
        sample: DatasetSchema,
        strategy: PromptStrategy,
        rag_enabled: bool,
        repeat_index: int,
        retrieval_snapshot: Optional[RetrievalSnapshot],
    ) -> RunRecord:
        """Run a single benchmark call gated by a concurrency semaphore."""
        async with semaphore:
            return await self._run_one(
                model=model,
                adapter=adapter,
                sample=sample,
                strategy=strategy,
                rag_enabled=rag_enabled,
                repeat_index=repeat_index,
                retrieval_snapshot=retrieval_snapshot,
            )

    async def run(self) -> Dict[str, object]:
        """
        Execute full benchmark matrix and return summary + record paths.

        Model calls within each (sample, strategy) pair are parallelized
        using asyncio.gather with a concurrency semaphore.
        """
        samples = self._load_all_samples()
        models = [m for m in self.config.models if m.enabled]
        conditions = self._condition_grid()
        logger.info(
            "Benchmark start: %d samples, %d models, %d conditions, %d repeats (max_concurrent=%d)",
            len(samples),
            len(models),
            len(conditions),
            self.config.repeats,
            self.config.max_concurrent,
        )

        adapters = {
            m.model_id: create_provider_adapter(m.provider, m.model_name, m.api_key_env)
            for m in models
        }

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        records: List[RunRecord] = []
        retrieval_cache: Dict[Tuple[str, str], RetrievalSnapshot] = {}

        for sample in samples:
            for strategy, rag_enabled in conditions:
                snapshot = None
                if rag_enabled and self.retriever:
                    key = (sample.sample_id, strategy.value)
                    if self.config.fair_shared_context and key in retrieval_cache:
                        snapshot = retrieval_cache[key]
                    else:
                        snapshot = self.retriever.retrieve(
                            query=sample.question,
                            query_id=sample.sample_id,
                            run_id=uuid.uuid4().hex,
                        )
                        if self.config.fair_shared_context:
                            retrieval_cache[key] = snapshot

                # Build all tasks for this (sample, strategy) pair
                tasks = []
                for repeat_idx in range(self.config.repeats):
                    for model in models:
                        tasks.append(
                            self._gated_run(
                                semaphore=semaphore,
                                model=model,
                                adapter=adapters[model.model_id],
                                sample=sample,
                                strategy=strategy,
                                rag_enabled=rag_enabled,
                                repeat_index=repeat_idx,
                                retrieval_snapshot=snapshot,
                            )
                        )

                batch = await asyncio.gather(*tasks)
                records.extend(batch)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ledger_path = output_dir / f"run_records_{stamp}.jsonl"
        with ledger_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

        summary = self._summarize(records)
        summary_path = output_dir / f"summary_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Benchmark complete. Records: %s, Summary: %s", ledger_path, summary_path)
        return {
            "records_path": str(ledger_path),
            "summary_path": str(summary_path),
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
            if track == DatasetTrack.MCQ.value:
                row["accuracy"] = sum(1 for r in valid if r.correct) / len(valid)
                row["partial_rate"] = (
                    sum(float(r.metrics.get("partial_match", 0.0)) for r in valid) / len(valid)
                )
            else:
                row["f1_mean"] = sum(float(r.metrics.get("f1", 0.0)) for r in valid) / len(valid)
                row["citation_match_rate"] = (
                    sum(float(r.metrics.get("citation_match", 0.0)) for r in valid) / len(valid)
                )
            table.append(row)

        return {
            "experiment": self.config.name,
            "seed": self.config.seed,
            "repeats": self.config.repeats,
            "rows": table,
        }


async def run_benchmark(config: ExperimentConfig, retriever: Optional[RetrieverAdapter] = None):
    """Convenience helper for script entrypoints."""
    runner = BenchmarkRunner(config=config, retriever=retriever)
    return await runner.run()

