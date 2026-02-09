# Nature-Grade Benchmark Registry (TPN-RAG-Public)

This document freezes the *canonical* experimental definitions used for the paper.
Any benchmark/figure/table claimed as a paper result must reference this registry
and the associated run manifests.

Last updated: 2026-02-08

## 1) Datasets (Frozen)

### MCQ124 (primary classification endpoint)
- Source: `data/eval/MCQ_Evaluation_Set_Final.xlsx`
- Canonical JSONL: `eval/data/benchmark_2026-02-05/mcq_holdout.jsonl`
- Split: `holdout`
- Expected N: 124
- Track: `mcq`

### Open-Ended QandA21 (clinical workflow tasks)
- Source: `data/eval/QandA_Evaluation_Set.xlsx`
- Canonical JSONL: `eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl`
- Split: `holdout`
- Expected N: 20 (1 row skipped in conversion: id=10 missing expected_answer)
- Track: `open_ended`

### Calc-50 (primary open-ended calc endpoint)
- Source: `data/eval/TPN_Calculation_QA_200.csv`
- Canonical JSONL: `eval/data/calc_50_holdout.jsonl`
- Manifest: `eval/data/calc_50_manifest.json`
- Split: `holdout`
- Expected N: 50
- Track: `open_ended`

### Calc-200 (supplemental)
- Source: `data/eval/TPN_Calculation_QA_200.csv`
- Canonical JSONL: `eval/data/calc_200_holdout.jsonl`
- Conversion manifest: `eval/data/calc_conversion_manifest.json`
- Split: `holdout`
- Expected N: 200
- Track: `open_ended`

## 2) Knowledge Base (KB) Regimes (Frozen)

Two KB regimes are used. **Paper main results use KB-clean only.**

### KB-clean (paper main)
- Goal: clinically grounded, leakage-safe.
- Excludes:
  - question banks / study guides (e.g., NeoReviews Q-bank)
  - any evaluation-derived documents
- Manifest: `data/kb_manifests/kb_clean.json`
- Persist directory (indexes): `persist/kb_clean/` (not committed)

### KB-max (appendix only)
- Goal: maximum accuracy (open-book).
- Includes all documents in `data/documents/` (including Q-banks).
- Manifest: `data/kb_manifests/kb_max.json`
- Persist directory (indexes): `persist/kb_max/` (not committed)

## 3) Retrieval Config (Frozen Defaults)

Unless a run manifest explicitly declares otherwise, the canonical retrieval config is:
- `top_k = 6`
- `candidate_k = 40`
- `max_context_chars = 6000`
- iterative retrieval: enabled
- iterations: `2`
- max decompositions: `3`

Retrieval is evaluated in a *deterministic* manner via precomputed snapshots:
- snapshots live under `eval/cache/` (not committed by default)
- each snapshot file includes fingerprints for dataset/KB/retrieval config

## 4) RAG Conditions (Frozen)

For each model × strategy:

1. **Baseline**: `no_rag`
   - retrieved context is not injected

2. **Naive RAG**: `rag_always`
   - retrieved context is always injected (gating disabled)

3. **Gated RAG**: `rag_gated`
   - inject retrieved context only when retrieval confidence passes thresholds

Canonical gating thresholds (unless overridden in run manifest):
- `rag_min_top_score = 0.62`
- `rag_min_returned_chunks = 2`
- `rag_min_context_chars = 200`

## 5) Prompting Strategies (Frozen)

### MCQ124
- `ZS`, `FEW_SHOT`, `COT` only
- Output constraint: final line must be `Answer: <letter(s)>`
- RAP is excluded from paper runs.

### Calc-50 / Calc-200
- Primary: `ZS`
- Diagnostic: `COT` (minimal work, not verbose)

### QandA21
- Primary: `ZS`
- The dataset question field includes the case context (converter embeds it).

## 6) Repeats / Randomness Policy

- Paper default: `repeats = 3` for providers/models that honor determinism/seed sufficiently.
- Otherwise: `repeats = 1`, and report uncertainty via confidence intervals over the 124 items.

Temperature policy:
- MCQ runs use `temperature = 0` (deterministic) unless a run manifest declares otherwise.

## 7) Primary Endpoints (Paper)

### MCQ
- Accuracy (proportion correct) + 95% CI
- Paired significance: McNemar test (baseline vs each RAG condition)
- Secondary: latency, error rate, and (for gated RAG) context-used rate

### Open-ended
- Calc: deterministic numeric/unit metrics (primary) + LLM-judge metrics (secondary)
- QandA21: hybrid (deterministic extraction where applicable + LLM-judge correctness)
- RAG-only: faithfulness + contextual precision/recall + citation compliance

## 8) Judge Policy (Open-Ended)

- Primary judge: `gpt-5-mini`
- Secondary judge: `gemini-3-flash` on a stratified 30% subset (agreement reporting)

Judge prompts/rubrics must be versioned and included in supplemental materials.

## 9) Artifact Layout (Paper Runs)

Paper run artifacts must be stored under:
- `eval/paper_runs/<run_set_id>/...`

Each run directory must include a `run_manifest.json` containing:
- git commit hash
- dataset fingerprint(s)
- KB manifest fingerprint
- retrieval config fingerprint (or snapshot meta fingerprint)
- prompt/template fingerprint
- model/provider details and decoding params
- RAG mode (`no_rag` / `rag_always` / `rag_gated`)
- context-used rate (for rag_gated)
