# QandA20 v2 Metrics Guide (DeepEval / GEval)

This note explains exactly what each score in the QandA20 v2 benchmark means, where it comes from in the code, and how we turn scores into PASS/FAIL.

## 1) What We Evaluate

Each QandA20 sample is an **open-ended clinical TPN question** with a **reference answer** (ground truth).

For each model we run three conditions:

- `no_rag`: prompt has **no retrieved context**.
- `rag_gated`: prompt includes retrieved context only if it passes a retrieval-quality gate.
- `rag_always`: prompt always includes retrieved context.

All runs are **zero-shot (ZS)**.

## 2) Output Contract (Why Some Rows Fail Fast)

Generation uses a strict output contract so that evaluation is deterministic and easy to audit:

- The model output must begin with **`Final answer:`**.
- The output must contain only the final answer text (no analysis/thinking/work).
- No citations/sources/document names.

If the model violates the contract, the runner does **one retry** with an explicit correction instruction. The resulting fields are:

- `format_ok` (bool)
- `format_retry_used` (bool)
- `format_violation_reason` / `format_violation_reason_after_retry`

Code: `/Users/chandra/Documents/TPN-RAG-Public/app/evaluation/format_metrics.py` and `/Users/chandra/Documents/TPN-RAG-Public/app/evaluation/benchmark_runner.py`.

## 3) DeepEval / GEval (LLM-as-Judge)

DeepEval produces per-sample metric scores by calling an **evaluation LLM** (the “judge”).

### 3.1 GEval Correctness (Primary)

**Metric:** `TPN_OpenCorrectness` (implemented via DeepEval `GEval`)

- Scale: 0.0 to 1.0
- Meaning: how clinically correct the answer is **relative to the reference answer**, allowing paraphrase.
- Why we use it: open-ended answers can be phrased many ways; keyword overlap is not reliable.

### 3.2 Answer Relevancy (Guardrail)

**Metric:** Answer relevancy

- Scale: 0.0 to 1.0
- Meaning: did the model answer the question that was asked.
- Why we use it: prevents rewarding “generally correct” but off-target responses.

### 3.3 RAG Diagnostics (Only When Retrieval Context Exists)

These are computed only for RAG conditions when `retrieval_context` exists.

- **Faithfulness:** is the model output supported by the retrieved context (hallucination check).
- **Contextual precision/recall/relevancy:** measures retrieval quality vs the reference answer.

Important: in this v2 run, we compute these diagnostics using the **primary judge (OpenAI)** for stability/cost, and we use the secondary judge mainly for agreement on correctness/relevancy.

## 4) Judges Used

This run includes two judges:

- OpenAI: `openai:gpt-4.1-mini-2025-04-14`
- Anthropic: `anthropic:claude-haiku-4-5-20251001`

Gemini was not included in the final v2 scoring run due to earlier JSON-schema instability in some DeepEval metric calls (it caused hard failures mid-run).

## 5) Pass / Partial / Fail Policy

Paper-facing grading is derived from the **primary judge** (OpenAI) plus the output contract:

- **PASS:** correctness ≥ 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- **PARTIAL:** 0.60 ≤ correctness < 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- **FAIL:** otherwise

Rationale:

- The 0.80 threshold is a “high confidence clinically correct” bar.
- The 0.60–0.80 band captures partially correct answers that still address the question.

## 6) Where To Find The Raw Records

Tracked paper artifacts:

- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/per_sample_review.csv`
- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/best_by_question.csv`
- `/Users/chandra/Documents/TPN-RAG-Public/paper/open_qanda20_v2/summary_by_model_condition.csv`

Full run artifacts (gitignored):

- Generation JSONL: `eval/paper_runs/open_qanda20_v2_20260209_200737/open/qanda20/<model>/<condition>/run_records_*.jsonl`
- DeepEval JSONL: `eval/paper_runs/open_qanda20_v2_20260209_200737/deepeval/open/qanda20/<model>/<condition>/<judge>/deepeval_records_*.jsonl`

