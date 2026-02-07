#!/usr/bin/env bash
set -euo pipefail

# Fast benchmark runner for GPT-5 mini (RAG vs no-RAG).
# Expected cwd: repo root, venv active, .env configured.

MCQ_DATASET="${MCQ_DATASET:-eval/data/benchmark_2026-02-05/mcq_holdout.jsonl}"
PERSIST_DIR="${PERSIST_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/benchmark_gpt5mini}"
REPEATS="${REPEATS:-3}"
TOP_K="${TOP_K:-6}"
CANDIDATE_K="${CANDIDATE_K:-40}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-6000}"
RETRIEVAL_ITERATIONS="${RETRIEVAL_ITERATIONS:-2}"
MAX_DECOMPOSITIONS="${MAX_DECOMPOSITIONS:-3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MODELS="${MODELS:-gpt-5-mini}"
RAG_GATING_DISABLED="${RAG_GATING_DISABLED:-0}"
RAG_MIN_TOP_SCORE="${RAG_MIN_TOP_SCORE:-0.62}"
RAG_MIN_RETURNED_CHUNKS="${RAG_MIN_RETURNED_CHUNKS:-2}"
RAG_MIN_CONTEXT_CHARS="${RAG_MIN_CONTEXT_CHARS:-200}"

echo "==> Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

echo "==> Running GPT-5 mini benchmark (baseline + RAG)"
GATING_FLAGS=()
if [[ "$RAG_GATING_DISABLED" == "1" ]]; then
  GATING_FLAGS+=(--disable-rag-gating)
fi
GATING_FLAGS+=(--rag-min-top-score "$RAG_MIN_TOP_SCORE")
GATING_FLAGS+=(--rag-min-returned-chunks "$RAG_MIN_RETURNED_CHUNKS")
GATING_FLAGS+=(--rag-min-context-chars "$RAG_MIN_CONTEXT_CHARS")

python scripts/run_benchmark.py \
  --mcq-dataset "$MCQ_DATASET" \
  --persist-dir "$PERSIST_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repeats "$REPEATS" \
  --top-k "$TOP_K" \
  --candidate-k "$CANDIDATE_K" \
  --max-context-chars "$MAX_CONTEXT_CHARS" \
  --retrieval-iterations "$RETRIEVAL_ITERATIONS" \
  --max-decompositions "$MAX_DECOMPOSITIONS" \
  "${GATING_FLAGS[@]}" \
  --models "$MODELS" \
  --max-concurrent "$MAX_CONCURRENT"

echo "==> Latest artifacts"
python scripts/tpnctl.py show-latest --output-dir "$OUTPUT_DIR"

echo "Done."
