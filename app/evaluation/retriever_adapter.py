"""
Deterministic retrieval adapter with fairness snapshots.

Standard retrieval behavior is now:
1) query decomposition
2) iterative retrieval/refinement
3) candidate fusion
4) final cross-encoder reranking

The flow is inspired by DeepRAG-like iterative retrieval while keeping
deterministic behavior for reproducible benchmark runs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..retrieval import CrossEncoderReranker, RRFConfig, RetrievalConfig, RetrievalPipeline
from .benchmark_types import (
    NormalizedChunk,
    RetrievalDiagnostics,
    RetrievalSnapshot,
    stable_text_hash,
)


@dataclass
class _CandidateDoc:
    """
    Lightweight candidate record for final reranking.

    CrossEncoderReranker expects candidate objects exposing `content`, `metadata`,
    and `score` attributes.
    """

    content: str
    metadata: Dict[str, object]
    score: float


class RetrieverAdapter:
    """
    Canonical retrieval interface used by benchmark runner.

    Guarantees:
    - deterministic per-query snapshots (cacheable)
    - normalized chunk schema
    - diagnostics for observability
    - shared context for fairness across model comparisons
    """

    def __init__(
        self,
        persist_dir: str | Path,
        top_k: int = 10,
        candidate_k: int = 40,
        max_context_chars: int = 12000,
        iterative_retrieval: bool = True,
        retrieval_iterations: int = 2,
        max_query_decompositions: int = 4,
    ):
        self.persist_dir = Path(persist_dir)
        self.top_k = max(1, top_k)
        self.candidate_k = max(self.top_k, candidate_k)
        self._retrieval_top_k = self.candidate_k
        self.max_context_chars = max_context_chars
        self.iterative_retrieval = iterative_retrieval
        self.retrieval_iterations = max(1, retrieval_iterations)
        self.max_query_decompositions = max(1, max_query_decompositions)
        self._cache: Dict[str, RetrievalSnapshot] = {}

        # High-accuracy default stack:
        # - hybrid dense+BM25 with weighted RRF
        # - no per-query rerank during planning loop
        # - one final rerank over fused candidates
        retrieval_pool = max(self._retrieval_top_k * 2, 40)
        rrf_config = RRFConfig(
            k=60,
            vector_k=retrieval_pool,
            bm25_k=retrieval_pool,
            vector_weight=0.6,
            bm25_weight=0.4,
            final_k=self._retrieval_top_k,
        )
        config = RetrievalConfig(
            enable_hyde=False,
            enable_multi_query=False,
            enable_reranking=False,
            rrf_config=rrf_config,
            final_top_k=self._retrieval_top_k,
        )
        self.pipeline = RetrievalPipeline.from_persisted(self.persist_dir, config=config)
        self.final_reranker = CrossEncoderReranker()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(text.split()).strip()

    def _decompose_query(self, query: str) -> List[str]:
        """
        Deterministically split and expand a clinical query into a compact plan.
        """
        base = self._normalize_whitespace(query)
        if not base:
            return []

        plan: List[str] = [base]
        seen = {base.lower()}

        # Clause decomposition for multi-part questions.
        split_pattern = re.compile(
            r"\s*(?:;|,|\band\b|\bthen\b|\bwhile\b|\bversus\b|\bvs\.?\b)\s*",
            flags=re.IGNORECASE,
        )
        for clause in split_pattern.split(base):
            candidate = self._normalize_whitespace(clause)
            if len(candidate.split()) < 4:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            plan.append(candidate)
            seen.add(key)
            if len(plan) >= self.max_query_decompositions:
                break

        lowered = base.lower()
        expansions: List[str] = []
        if any(token in lowered for token in ["dose", "dosage", "requirement", "mg/kg", "g/kg"]):
            expansions.append("recommended dosing range lower upper limits")
        if any(token in lowered for token in ["monitor", "lab", "follow-up", "surveillance"]):
            expansions.append("monitoring frequency thresholds safety parameters")
        if any(token in lowered for token in ["risk", "complication", "toxicity", "adverse"]):
            expansions.append("complications prevention contraindications management")

        for suffix in expansions:
            candidate = f"{base} {suffix}"
            key = candidate.lower()
            if key in seen:
                continue
            plan.append(candidate)
            seen.add(key)
            if len(plan) >= self.max_query_decompositions:
                break

        return plan[: self.max_query_decompositions]

    def _refine_queries(
        self,
        original_query: str,
        latest_batches: Sequence[Tuple[str, List[object]]],
        seen_queries: Sequence[str],
    ) -> List[str]:
        """
        Create iteration+1 refinement queries from section/source metadata hints.
        """
        seen = {item.lower() for item in seen_queries}
        refinements: List[str] = []

        section_hints: List[str] = []
        source_hints: List[str] = []
        for _, rows in latest_batches:
            for row in rows[:10]:
                metadata = dict(getattr(row, "metadata", {}) or {})
                section = str(metadata.get("section") or metadata.get("section_header") or "").strip()
                source = str(metadata.get("source") or "").strip()
                if section and section not in section_hints:
                    section_hints.append(section)
                if source and source not in source_hints:
                    source_hints.append(source)

        for section in section_hints[:2]:
            candidate = f"{original_query} section {section}"
            key = candidate.lower()
            if key not in seen:
                refinements.append(candidate)
                seen.add(key)

        for source in source_hints[:2]:
            candidate = f"{original_query} source {source}"
            key = candidate.lower()
            if key not in seen:
                refinements.append(candidate)
                seen.add(key)

        return refinements[:2]

    def _fuse_batches(
        self,
        original_query: str,
        batches: Sequence[Tuple[str, int, List[object]]],
    ) -> List[_CandidateDoc]:
        """
        Fuse all iteration/query batches into one candidate pool.
        """
        fused: Dict[str, Dict[str, object]] = {}
        original_key = original_query.strip().lower()

        for query_text, iteration_idx, rows in batches:
            is_original = query_text.strip().lower() == original_key
            query_weight = 1.0 if is_original else 0.85
            iteration_weight = 1.0 / (1.0 + 0.25 * iteration_idx)

            for rank_idx, row in enumerate(rows, 1):
                content = str(getattr(row, "content", "") or "")
                metadata = dict(getattr(row, "metadata", {}) or {})
                rerank_score = float(getattr(row, "rerank_score", 0.0) or 0.0)
                source = str(metadata.get("source", "unknown"))
                page = metadata.get("page") or metadata.get("page_num") or ""
                chunk_id = str(metadata.get("chunk_id", ""))
                key = stable_text_hash(f"{source}|{page}|{chunk_id}|{content[:256]}")

                rank_component = 1.0 / (10.0 + rank_idx)
                add_score = query_weight * iteration_weight * (0.7 * rank_component + 0.3 * max(rerank_score, 0.0))

                if key not in fused:
                    fused[key] = {
                        "content": content,
                        "metadata": metadata,
                        "score": 0.0,
                        "best_rerank_score": rerank_score,
                    }

                fused[key]["score"] = float(fused[key]["score"]) + add_score
                if rerank_score > float(fused[key]["best_rerank_score"]):
                    fused[key]["best_rerank_score"] = rerank_score

        ordered = sorted(
            fused.values(),
            key=lambda row: (float(row["score"]), float(row["best_rerank_score"])),
            reverse=True,
        )

        candidates: List[_CandidateDoc] = []
        for rank_idx, row in enumerate(ordered[: self._retrieval_top_k], 1):
            metadata = dict(row["metadata"])
            metadata["_fused_retrieval_rank"] = rank_idx
            metadata["_fused_retrieval_score"] = float(row["score"])
            candidates.append(
                _CandidateDoc(
                    content=str(row["content"]),
                    metadata=metadata,
                    score=float(row["score"]),
                )
            )
        return candidates

    def _iterative_retrieve(
        self,
        query: str,
    ) -> Tuple[List[dict], int, bool, List[str]]:
        """
        Run decomposition + iterative retrieval + fusion + final rerank.
        """
        query_plan = self._decompose_query(query) if self.iterative_retrieval else [query]
        if not query_plan:
            query_plan = [query]

        all_queries = list(query_plan)
        current_queries = list(query_plan)
        batches: List[Tuple[str, int, List[object]]] = []
        iterations_run = 0
        refinement_used = False

        for iteration_idx in range(self.retrieval_iterations):
            iterations_run += 1
            latest: List[Tuple[str, List[object]]] = []
            for current_query in current_queries:
                result = self.pipeline.retrieve(current_query, top_k=self._retrieval_top_k)
                rows = list(result.results)
                batches.append((current_query, iteration_idx, rows))
                latest.append((current_query, rows))

            if not self.iterative_retrieval:
                break
            if iteration_idx >= self.retrieval_iterations - 1:
                break

            refined = self._refine_queries(query, latest, all_queries)
            if not refined:
                break
            refinement_used = True
            for item in refined:
                if item not in all_queries:
                    all_queries.append(item)
            current_queries = refined

        fused_candidates = self._fuse_batches(query, batches)
        reranked = self.final_reranker.rerank(query, fused_candidates, top_k=self._retrieval_top_k)
        docs = [
            {
                "content": row.content,
                "metadata": row.metadata,
                "score": row.rerank_score,
                "retrieval_rank": row.original_rank,
                "rerank_score": row.rerank_score,
            }
            for row in reranked
        ]
        return docs, iterations_run, refinement_used, all_queries

    def _normalize_chunks(self, docs: List[dict]) -> List[NormalizedChunk]:
        chunks: List[NormalizedChunk] = []
        for idx, item in enumerate(docs, 1):
            metadata = item.get("metadata", {})
            chunks.append(
                NormalizedChunk(
                    doc_id=str(metadata.get("doc_id", metadata.get("source", ""))),
                    source=str(metadata.get("source", "unknown")),
                    page=metadata.get("page") or metadata.get("page_num"),
                    section=metadata.get("section") or metadata.get("section_header"),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                    rank=idx,
                    retrieval_rank=item.get("retrieval_rank") or metadata.get("_fused_retrieval_rank"),
                    rerank_score=item.get("rerank_score"),
                )
            )
        return chunks

    def _pack_context(self, chunks: List[NormalizedChunk]) -> str:
        """
        Diversity-aware context packing:
        - keep top-ranked chunks as primary signal
        - inject bounded source diversity early
        - then fill remaining budget by rank order
        """
        by_source: Dict[str, List[NormalizedChunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)

        best_per_source: List[NormalizedChunk] = []
        for source_chunks in by_source.values():
            best_per_source.append(min(source_chunks, key=lambda x: x.rank))
        best_per_source.sort(key=lambda x: x.rank)

        diversity_budget = max(3, min(self.top_k, 8))
        selected: List[NormalizedChunk] = best_per_source[:diversity_budget]

        selected_ids = {id(chunk) for chunk in selected}
        for chunk in sorted(chunks, key=lambda x: x.rank):
            if id(chunk) not in selected_ids:
                selected.append(chunk)
                selected_ids.add(id(chunk))

        parts: List[str] = []
        total_chars = 0
        for chunk in selected:
            part = f"[Source: {chunk.source}{f', p.{chunk.page}' if chunk.page else ''}]\n{chunk.content}\n"
            if total_chars + len(part) > self.max_context_chars:
                break
            parts.append(part)
            total_chars += len(part)
        return "\n---\n".join(parts)

    def retrieve(
        self,
        query: str,
        query_id: str,
        run_id: str,
        force_refresh: bool = False,
    ) -> RetrievalSnapshot:
        cache_key = stable_text_hash(
            f"{query_id}:{query}:{self.top_k}:{self.candidate_k}:{self.max_context_chars}:"
            f"{self.iterative_retrieval}:{self.retrieval_iterations}:{self.max_query_decompositions}"
        )
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]

        start = time.time()
        docs, iteration_count, refinement_used, query_plan = self._iterative_retrieve(query)
        retrieval_time_ms = (time.time() - start) * 1000

        chunks = self._normalize_chunks(docs)
        primary_chunks = chunks[: self.top_k]
        context = self._pack_context(primary_chunks)
        context_hash = stable_text_hash(context)

        diag = RetrievalDiagnostics(
            query=query,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=0.0,
            source_diversity=len({chunk.source for chunk in primary_chunks}),
            context_chars=len(context),
            candidate_count=len(chunks),
            returned_count=len(primary_chunks),
            iteration_count=iteration_count,
            refinement_used=refinement_used,
            query_plan=query_plan,
        )

        snapshot = RetrievalSnapshot(
            query_id=query_id,
            run_id=run_id,
            top_k=self.top_k,
            context_hash=context_hash,
            context_text=context,
            chunks=primary_chunks,
            diagnostics=diag,
        )
        self._cache[cache_key] = snapshot
        return snapshot
