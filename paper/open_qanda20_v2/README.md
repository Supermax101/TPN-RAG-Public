# QandA20 Open-Ended Benchmark v2 (Paper Artifacts)

This folder contains the paper-ready outputs for the QandA20 v2 open-ended benchmark.

## What To Read

- `REPORT.md`: narrative summary + rankings + example questions/answers.
- `METRICS.md`: definitions for every score and how PASS/FAIL is derived.

## What To Use For Analysis

- `per_sample_review.csv`: per-question, per-model, per-condition table including:
  - question, reference answer, model output
  - DeepEval/GEval scores (OpenAI + Claude)
  - PASS/PARTIAL/FAIL grade
  - RAG diagnostics (when applicable)
  - output-format contract flags

- `summary_by_model_condition.csv`: aggregated metrics by (model, condition).

- `best_by_question.csv`: one “best row” per question (highest primary-judge correctness).

## Figures

- `figures/`: quick PNGs for slides.
- `figures_nature/`: Nature-style PNGs plus one interactive HTML.

## Run Provenance

The complete raw run artifacts (generation JSONLs, DeepEval JSONLs, logs) live under:

- `eval/paper_runs/open_qanda20_v2_20260209_200737/`

That directory is gitignored by default to avoid huge commits.

