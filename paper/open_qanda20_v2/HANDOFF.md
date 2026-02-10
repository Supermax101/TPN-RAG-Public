# Handoff: QandA20 Open-Ended Benchmark v2 (TPN-RAG-Public)

This document is a guided handoff for a new engineer/agent taking over the QandA20 open-ended benchmark work.

## 0) Repo + Run Folder

- Repo root: `/Users/chandra/Documents/TPN-RAG-Public`
- Branch with all v2 work: `codex/openended-benchmarking`

**Completed v2 run folder (gitignored):**

- `/Users/chandra/Documents/TPN-RAG-Public/eval/paper_runs/open_qanda20_v2_20260209_200737/`

**Tracked paper deliverables (safe to share):**

- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/`

## 1) What Was The Goal

Paper-grade benchmarking for **QandA20 (N=20)** open-ended clinical TPN questions with:

- **Zero-shot only** prompting (ZS)
- Three conditions per model:
  - `no_rag`
  - `rag_gated`
  - `rag_always`
- **Deterministic retrieval** using precomputed retrieval snapshots
- **Strict output contract** so answers are easy to score and audit:
  - Must start with `Final answer:`
  - No citations, no chain-of-thought / analysis sections
- LLM-as-judge scoring via **DeepEval / GEval**

## 2) Dataset + Retrieval Snapshots

- Dataset JSONL (QandA20 holdout):
  - `/Users/chandra/Documents/TPN-RAG-Public/eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl`
- Retrieval snapshots (KB-clean, deterministic retrieval for the 20 questions):
  - `/Users/chandra/Documents/TPN-RAG-Public/eval/cache/retrieval_snapshots_kbclean_qanda20.jsonl`

Snapshots are used so RAG runs do **not** re-embed questions or drift over time.

## 3) Models Included (11 total)

**HF (local / open weights):**

- `microsoft/phi-4`
- `openai/gpt-oss-20b`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `google/medgemma-27b-text-it`
- `google/gemma-3-27b-it`

**API models:**

- `gpt-5-mini`
- `gpt-5.2`
- `claude-sonnet-4-5-20250929`
- `gemini-3-flash-preview`
- `grok-4-1-fast-reasoning`
- `kimi-k2.5`

Reporting convention:

- Kimi is API-served, but the boss requested we treat it as an **“open” model** in the open-vs-closed grouping.

## 4) Prompting / Output Contract v2

The point was to make evaluation easy and deterministic.

### 4.1 Template

- `/Users/chandra/Documents/TPN-RAG-Public/app/prompting/templates/open_zero_shot.txt`

Key features:

- Tells the model it will receive context but must not cite it.
- Explicitly requires the first line to start with `Final answer:`.

### 4.2 System prompts

- `/Users/chandra/Documents/TPN-RAG-Public/app/prompting/system_prompt.py`

Open-ended prompts were updated to:

- ban citations/sources
- ban visible chain-of-thought
- require `Final answer:` only

### 4.3 Automatic enforcement (validator + 1 retry)

- Validator: `/Users/chandra/Documents/TPN-RAG-Public/app/evaluation/format_metrics.py`
- Called from: `/Users/chandra/Documents/TPN-RAG-Public/app/evaluation/benchmark_runner.py`

Fields written into generation run_records:

- `format_ok`
- `format_retry_used`
- `format_violation_reason` (+ after retry)

This is important because some models (esp. certain HF instruct models) try to emit analysis.

## 5) DeepEval / GEval Scoring

### 5.1 Judges

Final v2 scoring used 2 judges due to earlier stability issues:

- Primary: OpenAI `openai:gpt-4.1-mini-2025-04-14`
- Secondary: Anthropic `anthropic:claude-haiku-4-5-20251001`

Gemini as judge was not included in the final run due to invalid-JSON / schema failures (DeepEval expects structured JSON for some metrics).

### 5.2 Metrics

- Primary endpoint: `GEval` correctness (`TPN_OpenCorrectness`, 0–1)
- Guardrail: Answer relevancy (0–1)
- RAG-only diagnostics (OpenAI judge): faithfulness + contextual precision/recall/relevancy

Grading (paper-facing):

- PASS: correctness ≥ 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- PARTIAL: 0.60 ≤ correctness < 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- FAIL otherwise

Full metric definitions are in:

- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/METRICS.md`

## 6) Where Results Live

### 6.1 Tracked (paper-ready)

Folder:

- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/`

Key files:

- `REPORT.md`: boss-friendly narrative + rankings + examples
- `per_sample_review.csv`: per (question, model, condition) with outputs + scores + PASS/FAIL
- `best_by_question.csv`: for each question, the best-performing row (by primary judge correctness)
- `summary_by_model_condition.csv`: aggregated metrics by model/condition
- `figures/`: quick PNGs for the deck
- `figures_nature/`: 6 Nature-grade PNGs + one interactive HTML

### 6.2 Full run artifacts (gitignored)

- `/Users/chandra/Documents/TPN-RAG-Public/eval/paper_runs/open_qanda20_v2_20260209_200737/`

Contains generation JSONLs, DeepEval per-judge JSONLs, logs, etc.

## 7) How To Regenerate The Report (Local)

Use the project venv (important: system python may have incompatible pandas/numpy):

```bash
cd /Users/chandra/Documents/TPN-RAG-Public
.venv/bin/python data_viz/open_qanda20_v2_report.py \
  --run-dir eval/paper_runs/open_qanda20_v2_20260209_200737 \
  --judges openai,anthropic \
  --paper-out paper/open_qanda20_v2
```

Nature-grade figures:

```bash
cd /Users/chandra/Documents/TPN-RAG-Public
.venv/bin/python data_viz/open_qanda20_v2_nature_figures.py \
  --run-dir eval/paper_runs/open_qanda20_v2_20260209_200737
```

## 7.1) How To Re-Run Generation + DeepEval (New VM)

The original Vast.ai VM used for v2 was destroyed, but the repo now has a
first-class orchestrator command that replaces ad-hoc queue scripts:

- Orchestrator CLI: `/Users/chandra/Documents/TPN-RAG-Public/scripts/tpnctl.py`
- Command: `paper-open-qanda20`

High-level behavior:

- HF models run **sequentially** (GPU-safe; avoids loading multiple large HF models at once).
- API models can run **in parallel** (CPU-bound; optional).
- DeepEval scoring runs after generation and writes into the same run folder.

Example (run everything, write into a fresh run folder):

```bash
cd /root/TPN-RAG-Public
source .venv/bin/activate
source .env  # must contain OPENAI_API_KEY, ANTHROPIC_API_KEY, (optional GEMINI_API_KEY), HF_TOKEN, etc.

python3 scripts/tpnctl.py paper-open-qanda20 \
  --run-set-id open_qanda20_v2_<timestamp> \
  --no-gemini \
  --run-api \
  --max-concurrent 5 \
  --deepeval-max-concurrent 5
```

Notes:

- If you want Gemini as a judge, remove `--no-gemini`, but expect occasional schema/JSON failures unless DeepEval prompting/parsing is hardened.
- RAG determinism requires the snapshot file to exist on the VM:
  - `eval/cache/retrieval_snapshots_kbclean_qanda20.jsonl`

## 8) What Needs Polish / Known Issues

1. **RAG lift is small on QandA20 overall** (`~+0.016` correctness mean on OpenAI judge). This is not necessarily “bad”, but it means the deck narrative must emphasize:
   - per-model lift (HF tends to benefit more)
   - per-question examples where lift is large
   - retrieval metrics (context recall bottleneck)

2. **Format contract failures for some HF models**:
   - `gpt-oss-20b` and `gemma3-27b` have very low `format_ok_rate` in this run.
   - If needed, strengthen the retry prompt or add a structured-output fallback for open-ended.

3. **Gemini as judge**:
   - Earlier DeepEval runs failed because some metric prompts expect strict JSON and Gemini sometimes produced invalid JSON.
   - If Gemini is required later, we should harden `deepeval_open_eval.py` with more aggressive JSON repair or force Gemini response format.

## 9) VM

The Vast.ai VM used for generation/scoring was destroyed after completion. All needed artifacts are local.
