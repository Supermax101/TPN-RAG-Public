# Nature-Grade Benchmark Registry (TPN-RAG-Public)

This document freezes the *canonical* experimental definitions used for the paper.
Any benchmark/figure/table claimed as a paper result must reference this registry
and the associated run manifests.

Last updated: 2026-02-10

## 1) Datasets (Frozen)

### MCQ124 (primary classification endpoint)
- Source: `data/eval/MCQ_Evaluation_Set_Final.xlsx`
- Canonical JSONL: `eval/data/benchmark_2026-02-05/mcq_holdout.jsonl`
- Split: `holdout`
- Expected N: 124
- Track: `mcq`

### Open-Ended QandA20 (clinical workflow tasks)
- Source: `data/eval/QandA_Evaluation_Set.xlsx`
- Canonical JSONL: `eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl`
- Split: `holdout`
- Expected N: 20 (1 row skipped in conversion: id=10 missing expected_answer)
- Track: `open_ended`

### Takeoff41-200 (nutrition QA, mixed)
- Source: `data/eval/takeoff41_200_tpn_nutrition_qa.jsonl`
- Canonical JSONL: `eval/data/benchmark_2026-02-05/takeoff41_200_holdout.jsonl`
- Conversion manifest: `eval/data/benchmark_2026-02-05/takeoff41_200_conversion_manifest.json`
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

### Open-ended (QandA20 + Takeoff41-200)
- Paper main: `ZS` only
- The dataset question field includes the case context when available (converter embeds it).
- Output contract (paper-grade runs): response must start with `Final answer:` and must not include citations or chain-of-thought.

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
- Primary: LLM-judge correctness (reference-based, paraphrase-tolerant)
- Secondary: deterministic extraction metrics (F1, key phrase overlap, numeric/unit extraction where applicable)
- RAG-only: faithfulness + contextual precision/recall + citation compliance

## 8) Judge Policy (Open-Ended)

- Primary judge (full coverage): `openai:gpt-4.1-mini-2025-04-14`
- Secondary judge (agreement/sensitivity reporting): `anthropic:claude-haiku-4-5-20251001`

Notes:
- We attempted a tri-judge setup including Gemini, but Gemini sometimes produced invalid JSON for schema-based metric prompts, causing hard failures mid-run. Gemini can be re-enabled once `deepeval_open_eval.py` is hardened for JSON repair or stricter response formatting.
- `gpt-5-mini` was not used as a judge because it produced frequent safety/policy refusals on neonatal/TPN clinical content, which makes paper-grade scoring unstable.

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
