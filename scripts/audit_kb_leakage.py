#!/usr/bin/env python3
"""
Audit evaluation-set leakage against the KB.

This is NOT split-leakage (train vs holdout); this is "is the holdout question
present in the retrieved KB text?" which can happen if question banks/study
guides are ingested.

Approach:
- Use persisted BM25 corpus (chunk text) as the searchable KB surface.
- For each eval question, retrieve top-N candidate chunks via BM25.
- Compute:
  - exact substring match (normalized)
  - fuzzy 5-gram Jaccard similarity

Outputs:
- JSON report with suspected leaks and top matching chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _question_only(question: str) -> str:
    # Converter embeds case context. For leakage we care about the core question.
    marker = "\n\nquestion:\n"
    q = question or ""
    lower = q.lower()
    idx = lower.find(marker)
    if idx >= 0:
        return q[idx + len(marker) :].strip()
    return q.strip()


def _char_ngrams(s: str, n: int = 5) -> set[str]:
    s = _normalize_text(s).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(0, len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit eval dataset leakage against KB (BM25 corpus)")
    p.add_argument("--dataset", type=str, required=True, help="Eval dataset JSONL path (mcq_holdout.jsonl, etc.)")
    p.add_argument(
        "--persist-dir",
        type=str,
        default="data",
        help="Persist dir containing bm25/corpus.json + tokenized.json + metadata.json",
    )
    p.add_argument("--topk", type=int, default=50, help="BM25 top-k candidates to consider per question")
    p.add_argument("--report-topn", type=int, default=5, help="How many top matches to store per sample")
    p.add_argument("--fuzzy-threshold", type=float, default=0.75, help="Flag as suspected leak if Jaccard >= threshold")
    p.add_argument("--out", type=str, default="", help="Output JSON path (default: eval/results/leakage_audit_*.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = (PROJECT_ROOT / persist_dir).resolve()

    corpus_path = persist_dir / "bm25" / "corpus.json"
    tokenized_path = persist_dir / "bm25" / "tokenized.json"
    metadata_path = persist_dir / "bm25" / "metadata.json"
    for pth in [corpus_path, tokenized_path, metadata_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing BM25 artifact: {pth}")

    rows = _load_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"Empty dataset: {dataset_path}")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tokenized = json.loads(tokenized_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not (len(corpus) == len(tokenized) == len(metadata)):
        raise SystemExit("BM25 corpus/tokenized/metadata length mismatch")

    from rank_bm25 import BM25Okapi
    from app.retrieval.tokenizer import clinical_tokenize

    bm25 = BM25Okapi(tokenized)

    suspected: List[dict] = []
    per_sample: List[dict] = []

    started = time.time()
    for idx, r in enumerate(rows, 1):
        sid = str(r.get("sample_id") or f"row_{idx}")
        q_full = str(r.get("question") or "")
        q = _question_only(q_full)
        q_norm = _normalize_text(q)
        q_grams = _char_ngrams(q, n=5)

        query_tokens = clinical_tokenize(q)
        scores = bm25.get_scores(query_tokens)
        # TopK indices by score (descending).
        # Note: BM25Okapi returns a numpy array-like list; sort indices.
        topk = int(args.topk)
        top_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:topk]

        matches: List[dict] = []
        best = {"fuzzy": 0.0, "exact": False}

        for i in top_idx:
            text = str(corpus[i] or "")
            src = str(metadata[i].get("source") or "unknown")
            chunk_index = metadata[i].get("chunk_index")

            text_norm = _normalize_text(text)
            exact = bool(q_norm) and (q_norm in text_norm or text_norm in q_norm)
            fuzzy = _jaccard(q_grams, _char_ngrams(text, n=5))

            if exact or fuzzy >= max(best["fuzzy"], 0.0):
                best = {"fuzzy": float(fuzzy), "exact": bool(exact), "source": src, "chunk_index": chunk_index}

            matches.append(
                {
                    "source": src,
                    "chunk_index": chunk_index,
                    "bm25_score": float(scores[i]),
                    "exact_substring": bool(exact),
                    "fuzzy_jaccard_5gram": float(fuzzy),
                    "chunk_preview": text[:240],
                }
            )

        matches_sorted = sorted(matches, key=lambda m: (m["exact_substring"], m["fuzzy_jaccard_5gram"], m["bm25_score"]), reverse=True)
        topn = matches_sorted[: int(args.report_topn)]

        is_suspected = bool(best.get("exact")) or float(best.get("fuzzy") or 0.0) >= float(args.fuzzy_threshold)
        sample_row = {
            "sample_id": sid,
            "track": str(r.get("track") or ""),
            "question_preview": q[:240],
            "best_match": best,
            "suspected_leak": bool(is_suspected),
            "top_matches": topn,
        }
        per_sample.append(sample_row)
        if is_suspected:
            suspected.append(sample_row)

    elapsed_s = time.time() - started

    report = {
        "dataset_path": str(dataset_path),
        "persist_dir": str(persist_dir),
        "bm25_corpus_size": len(corpus),
        "params": {
            "topk": int(args.topk),
            "report_topn": int(args.report_topn),
            "fuzzy_threshold": float(args.fuzzy_threshold),
        },
        "summary": {
            "total_samples": len(rows),
            "suspected_leaks": len(suspected),
            "elapsed_s": round(elapsed_s, 3),
        },
        "suspected": suspected,
        "per_sample": per_sample,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (PROJECT_ROOT / "eval" / "results" / f"leakage_audit_{stamp}.json")
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved leakage audit: {out_path}")
    print(f"Suspected leaks: {len(suspected)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

