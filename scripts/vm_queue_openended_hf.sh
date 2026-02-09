#!/usr/bin/env bash
set -euo pipefail

# Queue open-ended benchmark runs for HuggingFace (local) models.
#
# This script is designed for long-running VM sessions:
# - One model per process (avoids holding multiple HF models in GPU memory).
# - Three paper RAG conditions per model:
#   1) no_rag      (baseline only)
#   2) rag_gated   (RAG only, gating enabled)
#   3) rag_always  (RAG only, gating disabled)
#
# Expected:
# - repo root as cwd
# - venv active (or run with `python3 ...`)
# - KB-clean already ingested under $PERSIST_DIR (contains bm25/ + chromadb/ + ingestion_manifest.json)
# - retrieval snapshots already computed for each dataset (see $QANDA_SNAPSHOTS / $TAKEOFF_SNAPSHOTS)

RUN_SET_ID="${RUN_SET_ID:-open_hf_$(date +%Y%m%d_%H%M%S)}"
PERSIST_DIR="${PERSIST_DIR:-persist/kb_clean}"

# Datasets
QANDA_DATASET="${QANDA_DATASET:-eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl}"
TAKEOFF_DATASET="${TAKEOFF_DATASET:-eval/data/benchmark_2026-02-05/takeoff41_200_holdout.jsonl}"

# Snapshot inputs (for RAG runs only)
QANDA_SNAPSHOTS="${QANDA_SNAPSHOTS:-eval/cache/retrieval_snapshots_kbclean_qanda20.jsonl}"
TAKEOFF_SNAPSHOTS="${TAKEOFF_SNAPSHOTS:-eval/cache/retrieval_snapshots_kbclean_takeoff41_200.jsonl}"

# Paper-main open-ended strategy
STRATEGIES="${STRATEGIES:-ZS}"
REPEATS="${REPEATS:-1}"

# HF model keys (must match scripts/run_benchmark.py --list-models keys)
MODELS="${MODELS:-phi-4,gpt-oss-20b,qwen3-30b-a3b,medgemma-27b,gemma3-27b}"

# If set to 1, continue to the next model even if one run fails.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

OUTPUT_ROOT="eval/paper_runs/${RUN_SET_ID}/open"
LOG_ROOT="eval/paper_runs/${RUN_SET_ID}/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

echo "==> Run set: $RUN_SET_ID"
echo "==> Models:  $MODELS"
echo "==> Strategy(s): $STRATEGIES  repeats=$REPEATS"
echo "==> Persist dir: $PERSIST_DIR"
echo "==> Output root: $OUTPUT_ROOT"

run_one () {
  local dataset_tag="$1"     # qanda20 | takeoff41_200
  local dataset_path="$2"
  local snapshots_path="$3" # may be empty for no_rag
  local model_key="$4"
  local condition="$5"       # no_rag | rag_gated | rag_always

  local out_dir="${OUTPUT_ROOT}/${dataset_tag}/${model_key}/${condition}"
  local log_path="${LOG_ROOT}/${dataset_tag}_${model_key}_${condition}.log"
  mkdir -p "$out_dir"

  echo "==> [$dataset_tag] [$model_key] [$condition]"
  echo "    dataset:   $dataset_path"
  echo "    snapshots: ${snapshots_path:-<none>}"
  echo "    out_dir:   $out_dir"
  echo "    log:       $log_path"

  local args=(
    python scripts/run_benchmark.py
    --open-dataset "$dataset_path"
    --persist-dir "$PERSIST_DIR"
    --output-dir "$out_dir"
    --models "$model_key"
    --strategies "$STRATEGIES"
    --repeats "$REPEATS"
  )

  if [[ "$condition" == "no_rag" ]]; then
    args+=(--no-rag)
  else
    if [[ -z "$snapshots_path" ]]; then
      echo "ERROR: snapshots_path is required for $condition" >&2
      return 2
    fi
    args+=(--no-baseline --retrieval-snapshots-in "$snapshots_path")
    if [[ "$condition" == "rag_always" ]]; then
      args+=(--disable-rag-gating)
    fi
  fi

  # Note: pipefail is enabled; failures propagate correctly through tee.
  "${args[@]}" 2>&1 | tee "$log_path"
}

IFS=',' read -r -a MODEL_KEYS <<< "$MODELS"

for model_key in "${MODEL_KEYS[@]}"; do
  model_key="$(echo "$model_key" | xargs)"
  if [[ -z "$model_key" ]]; then
    continue
  fi

  for dataset_tag in "qanda20" "takeoff41_200"; do
    if [[ "$dataset_tag" == "qanda20" ]]; then
      dataset_path="$QANDA_DATASET"
      snapshots_path="$QANDA_SNAPSHOTS"
    else
      dataset_path="$TAKEOFF_DATASET"
      snapshots_path="$TAKEOFF_SNAPSHOTS"
    fi

    # Baseline (no_rag)
    if ! run_one "$dataset_tag" "$dataset_path" "" "$model_key" "no_rag"; then
      [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
    fi

    # RAG gated
    if ! run_one "$dataset_tag" "$dataset_path" "$snapshots_path" "$model_key" "rag_gated"; then
      [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
    fi

    # RAG always (gating disabled)
    if ! run_one "$dataset_tag" "$dataset_path" "$snapshots_path" "$model_key" "rag_always"; then
      [[ "$CONTINUE_ON_ERROR" == "1" ]] || exit 1
    fi
  done
done

echo "==> Queue complete. Outputs under: $OUTPUT_ROOT"

