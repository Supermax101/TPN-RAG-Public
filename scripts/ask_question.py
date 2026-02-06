#!/usr/bin/env python3
"""
Helper module for ask / quick-test CLI commands.

Provides a single ``ask_one()`` coroutine that wires up:
- provider adapter (any supported provider)
- optional RetrieverAdapter for RAG
- PromptRenderer with chosen strategy
- TPN specialist system prompt
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.evaluation.provider_adapter import create_provider_adapter
from app.evaluation.retriever_adapter import RetrieverAdapter
from app.evaluation.prompting import render_prompt
from app.evaluation.benchmark_types import PromptStrategy
from app.prompting import get_system_prompt

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250514",
    "gemini": "gemini-2.5-flash",
    "xai": "grok-4-fast-reasoning",
    "kimi": "kimi-k2-0905-preview",
}


async def ask_one(
    question: str,
    provider: str = "openai",
    model: Optional[str] = None,
    strategy: str = "ZS",
    with_rag: bool = False,
    persist_dir: str = "./data",
) -> dict:
    """
    Ask a single question and return structured result.

    Returns dict with keys: answer, context, strategy, provider, model, latency_ms
    """
    model_name = model or DEFAULT_MODELS.get(provider, "gpt-4o")
    adapter = create_provider_adapter(provider, model_name)

    strat = PromptStrategy(strategy.upper())

    context_text = None
    retrieval_info = None
    if with_rag:
        retriever = RetrieverAdapter(
            persist_dir=persist_dir,
            top_k=6,
            candidate_k=40,
            max_context_chars=6000,
            max_query_decompositions=3,
        )
        snapshot = retriever.retrieve(query=question, query_id="cli", run_id="cli")
        context_text = snapshot.context_text
        retrieval_info = {
            "chunks": len(snapshot.chunks),
            "sources": list({c.source for c in snapshot.chunks}),
            "context_chars": len(context_text),
        }

    prompt = render_prompt(
        strategy=strat,
        question=question,
        options=None,
        context=context_text,
    )

    result = await adapter.generate(
        prompt=prompt,
        system=get_system_prompt(use_rag=with_rag),
        temperature=0.0,
        max_tokens=1000,
        model_id=model_name,
    )

    return {
        "answer": result.text,
        "context": context_text,
        "retrieval": retrieval_info,
        "strategy": strat.value,
        "provider": provider,
        "model": model_name,
        "latency_ms": result.latency_ms,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask a TPN question")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--provider", default="openai", help="Provider name")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--strategy", default="ZS", help="Prompt strategy")
    parser.add_argument("--with-rag", action="store_true", help="Enable RAG")
    parser.add_argument("--persist-dir", default="./data", help="Retrieval index dir")
    args = parser.parse_args()

    result = asyncio.run(ask_one(
        question=args.question,
        provider=args.provider,
        model=args.model,
        strategy=args.strategy,
        with_rag=args.with_rag,
        persist_dir=args.persist_dir,
    ))

    print(f"\nProvider: {result['provider']} / {result['model']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    if result['retrieval']:
        print(f"RAG: {result['retrieval']['chunks']} chunks from {result['retrieval']['sources']}")
    print(f"\n{'='*60}")
    print(result['answer'])
