"""
Agentic retrieval adapter with LLM-based relevance judging.

Implements the Self-RAG / CRAG pattern:
1. Retrieve with base RetrieverAdapter
2. LLM judges each chunk for relevance
3. If too few relevant → rewrite query and re-retrieve
4. Return updated RetrievalSnapshot (same type, benchmark runner unchanged)

Opt-in via ``agentic_retrieval=True`` in ExperimentConfig.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import List, Optional

from .benchmark_types import (
    NormalizedChunk,
    RetrievalDiagnostics,
    RetrievalSnapshot,
    stable_text_hash,
)
from .provider_adapter import create_provider_adapter
from .retriever_adapter import RetrieverAdapter

logger = logging.getLogger(__name__)

# Prompts adapted from app/chains/agentic_rag.py
_GRADE_PROMPT = """You are a clinical document relevance grader for TPN (Total Parenteral Nutrition) questions.

Assess whether the retrieved document contains information relevant to answer the clinical question.

**Retrieved Document:**
{context}

**Clinical Question:**
{question}

Grade as "yes" if the document contains specific information helpful for answering this question.
Grade as "no" if the document is irrelevant or only tangentially related.

Respond with ONLY a JSON object: {{"score": "yes"}} or {{"score": "no"}}"""

_REWRITE_PROMPT = """You are a clinical query optimizer for TPN (Total Parenteral Nutrition) information retrieval.

The original query did not retrieve relevant documents. Rewrite it to be more specific and likely to find relevant TPN clinical information.

**Original Question:**
{question}

**Tips for rewriting:**
- Add specific TPN terminology (amino acids, dextrose, lipids, electrolytes)
- Include patient context (preterm, neonatal, pediatric)
- Mention ASPEN guidelines if relevant
- Focus on the core clinical concept

Respond with ONLY the rewritten query (one line, no explanation):"""


class AgenticRetrieverAdapter:
    """Wraps RetrieverAdapter with LLM-based relevance judging."""

    def __init__(
        self,
        base_retriever: RetrieverAdapter,
        judge_provider: str = "openai",
        judge_model: str = "gpt-4o-mini",
        judge_api_key_env: Optional[str] = None,
        relevance_threshold: float = 0.5,
    ):
        self.base_retriever = base_retriever
        self.relevance_threshold = relevance_threshold
        self._judge_adapter = create_provider_adapter(
            judge_provider, judge_model, judge_api_key_env
        )

    async def _judge_chunk(self, question: str, chunk: NormalizedChunk) -> bool:
        """Judge a single chunk for relevance. Returns True if relevant."""
        prompt = _GRADE_PROMPT.format(
            context=chunk.content[:2000], question=question
        )
        try:
            result = await self._judge_adapter.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=50,
            )
            text = result.text.strip().lower()
            # Parse JSON or free-text response
            if '"yes"' in text or "'yes'" in text:
                return True
            if '"no"' in text or "'no'" in text:
                return False
            # Fallback: look for yes/no
            return "yes" in text
        except Exception as e:
            logger.warning("Agentic judge call failed: %s", e)
            return True  # Fail-open: keep chunk if judge errors

    async def _judge_relevance(
        self, question: str, chunks: List[NormalizedChunk]
    ) -> List[NormalizedChunk]:
        """Grade all chunks concurrently, return only relevant ones."""
        tasks = [self._judge_chunk(question, chunk) for chunk in chunks]
        verdicts = await asyncio.gather(*tasks)
        return [c for c, relevant in zip(chunks, verdicts) if relevant]

    async def _rewrite_query(self, question: str) -> str:
        """Use LLM to rewrite query for better retrieval."""
        prompt = _REWRITE_PROMPT.format(question=question)
        try:
            result = await self._judge_adapter.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
            rewritten = result.text.strip()
            # Take first line only
            rewritten = rewritten.split("\n")[0].strip()
            if len(rewritten) > 10:
                return rewritten
        except Exception as e:
            logger.warning("Agentic query rewrite failed: %s", e)
        return question

    def retrieve(
        self,
        query: str,
        query_id: str,
        run_id: str,
        force_refresh: bool = False,
    ) -> RetrievalSnapshot:
        """
        Retrieve with LLM relevance judging.

        Synchronous interface matching RetrieverAdapter.retrieve().
        Internally runs async judge calls via asyncio.run().
        """
        # Step 1: Base retrieval
        snapshot = self.base_retriever.retrieve(
            query=query, query_id=query_id, run_id=run_id, force_refresh=force_refresh
        )

        # Step 2: LLM judges each chunk
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — use thread to avoid nested event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                relevant_chunks = pool.submit(
                    asyncio.run,
                    self._judge_relevance(query, snapshot.chunks),
                ).result()
        else:
            relevant_chunks = asyncio.run(
                self._judge_relevance(query, snapshot.chunks)
            )

        # Step 3: If <50% relevant, rewrite and re-retrieve
        if len(relevant_chunks) < len(snapshot.chunks) * self.relevance_threshold:
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    new_query = pool.submit(
                        asyncio.run, self._rewrite_query(query)
                    ).result()
            else:
                new_query = asyncio.run(self._rewrite_query(query))

            snapshot2 = self.base_retriever.retrieve(
                query=new_query,
                query_id=query_id,
                run_id=f"{run_id}_rewrite",
                force_refresh=True,
            )

            # Merge: relevant from original + all from rewrite, deduplicate
            seen_hashes = set()
            merged: List[NormalizedChunk] = []
            for chunk in relevant_chunks + snapshot2.chunks:
                h = stable_text_hash(chunk.content[:256])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    merged.append(chunk)

            # Re-rank by original rank, take top_k
            merged.sort(key=lambda c: c.rank)
            final_chunks = merged[: snapshot.top_k]

            # Rebuild context
            context_parts = []
            total = 0
            for chunk in final_chunks:
                part = f"[Source: {chunk.source}{f', p.{chunk.page}' if chunk.page else ''}]\n{chunk.content}\n"
                if total + len(part) > self.base_retriever.max_context_chars:
                    break
                context_parts.append(part)
                total += len(part)
            context_text = "\n---\n".join(context_parts)

            # Update diagnostics
            diag = snapshot.diagnostics.model_copy()
            diag.refinement_used = True
            diag.query_plan = diag.query_plan + [f"[agentic rewrite] {new_query}"]

            return RetrievalSnapshot(
                query_id=query_id,
                run_id=run_id,
                top_k=snapshot.top_k,
                context_hash=stable_text_hash(context_text),
                context_text=context_text,
                chunks=final_chunks,
                diagnostics=diag,
            )

        # If enough relevant, filter but keep original structure
        if len(relevant_chunks) < len(snapshot.chunks):
            context_parts = []
            total = 0
            for chunk in relevant_chunks:
                part = f"[Source: {chunk.source}{f', p.{chunk.page}' if chunk.page else ''}]\n{chunk.content}\n"
                if total + len(part) > self.base_retriever.max_context_chars:
                    break
                context_parts.append(part)
                total += len(part)
            context_text = "\n---\n".join(context_parts)

            return RetrievalSnapshot(
                query_id=query_id,
                run_id=run_id,
                top_k=snapshot.top_k,
                context_hash=stable_text_hash(context_text),
                context_text=context_text,
                chunks=relevant_chunks,
                diagnostics=snapshot.diagnostics,
            )

        return snapshot
