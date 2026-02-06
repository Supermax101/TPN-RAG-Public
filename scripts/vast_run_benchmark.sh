#!/usr/bin/env bash
set -euo pipefail

# Run benchmark end-to-end on remote GPU machine.
# Expected cwd: repo root, venv active, .env configured.

MCQ_DATASET="${MCQ_DATASET:-eval/data/benchmark_2026-02-05/mcq_holdout.jsonl}"
OPEN_DATASET="${OPEN_DATASET:-eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl}"
PERSIST_DIR="${PERSIST_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/benchmark}"
REPEATS="${REPEATS:-5}"
RETRIEVAL_CANDIDATE_K="${RETRIEVAL_CANDIDATE_K:-60}"
RETRIEVAL_ITERATIONS="${RETRIEVAL_ITERATIONS:-2}"
MAX_DECOMPOSITIONS="${MAX_DECOMPOSITIONS:-4}"
MODELS="${MODELS:-gpt-4o,claude-sonnet,gemini-2.5-pro,grok-4,kimi-k2}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-openai}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"

echo "==> Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

echo "==> Step 0: Convert Excel to JSONL"
python scripts/convert_eval_xlsx.py --out-dir "$(dirname "$MCQ_DATASET")"

echo "==> Step 1: Build indexes"
python scripts/tpnctl.py ingest \
  --docs-dir data/documents \
  --persist-dir "$PERSIST_DIR" \
  --embedding-provider "$EMBEDDING_PROVIDER" \
  --embedding-model "$EMBEDDING_MODEL"

echo "==> Step 2: Run benchmark matrix"
python scripts/tpnctl.py benchmark \
  --mcq-dataset "$MCQ_DATASET" \
  --open-dataset "$OPEN_DATASET" \
  --persist-dir "$PERSIST_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repeats "$REPEATS" \
  --candidate-k "$RETRIEVAL_CANDIDATE_K" \
  --retrieval-iterations "$RETRIEVAL_ITERATIONS" \
  --max-decompositions "$MAX_DECOMPOSITIONS" \
  --models "$MODELS"

echo "==> Step 3: Analyze outputs"
python scripts/tpnctl.py analyze \
  --output-dir "$OUTPUT_DIR" \
  --output-name "analysis_report.json"

echo "==> Step 4: Show latest artifacts"
python scripts/tpnctl.py show-latest --output-dir "$OUTPUT_DIR"

echo "Done."
