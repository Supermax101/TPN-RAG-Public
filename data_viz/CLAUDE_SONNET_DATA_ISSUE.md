# Claude Sonnet 4.5 — Benchmark Data Issue

**Date:** 2026-02-07  
**Status:** ⚠️ Excluded from charts pending re-run  
**Affected run:** `eval/results/benchmark_api_remaining_mcq124/claude-sonnet/`

---

## Summary

Claude Sonnet 4.5 benchmark results are **unreliable** due to a systematic answer-parsing failure in the MCQ evaluation pipeline. The model's Chain-of-Thought (CoT) responses are frequently **truncated** (hitting the max token limit before producing a final answer), and even complete responses often use answer formats that the `parse_mcq_response()` parser cannot extract.

The model is **excluded from all 10 publication charts** until a clean re-run is completed.

---

## Root Cause

The `parse_mcq_response()` function in `app/parsers/mcq_parser.py` uses a 7-priority regex extraction system expecting patterns like:
- `Answer: X`
- `\boxed{X}`
- `therefore X`
- Standalone letter on last line

Claude Sonnet 4.5's CoT responses are **verbose clinical reasoning** that often:
1. **Get truncated by max_tokens** — the model writes detailed multi-step reasoning and runs out of tokens before stating the final answer letter
2. **Use non-standard answer patterns** — e.g., "making option A the most appropriate", "Option C provides the best...", ending mid-sentence

The structured output rescue (Anthropic `generate_structured`) also fails for many of these, resulting in `parsed_answer = ""` (empty).

---

## Impact by Configuration

| Config | Reported Accuracy | Empty Parsed Answers | True Accuracy (est.) |
|--------|------------------|---------------------|---------------------|
| **Few-Shot No RAG** | 83.9% | 0 / 124 | ~83% ✅ (reliable) |
| **Few-Shot + RAG** | 87.1% | 5 / 124 | ~86–87% (mostly reliable) |
| **Zero-Shot No RAG** | 79.0% | 11 / 124 | ~77–79% (partially affected) |
| **Zero-Shot + RAG** | 78.2% | 19 / 124 | ~79% (partially affected) |
| **CoT No RAG** | **52.4%** 🚨 | **47 / 124** | Unknown — too many failures |
| **CoT + RAG** | **66.9%** 🚨 | **33 / 124** | Unknown — too many failures |

**Key observation:** Few-Shot is least affected (0 empty) because the examples teach Claude the expected `Answer: X` output format. CoT is most affected because the lengthy reasoning exhausts the token budget.

---

## Lenient Re-parsing Attempt

A lenient re-parser with additional patterns (e.g., "Option X is the most appropriate", "making option X the best") was applied but only rescued **12 additional correct answers** out of 75 empty parses:

| Config | Original | Lenient | Rescued |
|--------|----------|---------|---------|
| COT no_rag | 52.4% | 57.3% | +7 |
| COT rag | 66.9% | 68.5% | +3 |
| ZS rag | 78.2% | 79.0% | +2 |

**Conclusion:** Lenient re-parsing is insufficient. A re-run with increased `max_tokens` and/or an improved prompt suffix (forcing Claude to always end with `Answer: X`) is needed.

---

## Recommended Fix (for re-run)

1. **Increase `mcq_max_tokens`** for the Anthropic provider (currently likely 1024 or 2048 — Claude needs more for CoT)
2. **Add explicit format enforcement** to the CoT prompt suffix, e.g.:
   ```
   CRITICAL: After your reasoning, you MUST end with exactly:
   Answer: <letter>
   ```
3. **Fix parser bug** — Add these patterns to `parse_mcq_response()`:
   - `making (?:option )?([A-F]) (?:the )?(?:best|correct|most)` (Priority 4f)
   - `Option ([A-F]) (?:is )?(?:the )?(?:most|best|correct|appropriate)` (Priority 4g)
   - `the answer (?:would|should) be ([A-F])` (Priority 4h)

---

## Files Involved

- **Parser:** `app/parsers/mcq_parser.py` → `parse_mcq_response()` 
- **Runner:** `app/evaluation/benchmark_runner.py` → lines 376–430 (parse + retry + structured rescue)
- **Raw data:** `eval/results/benchmark_api_remaining_mcq124/claude-sonnet/run_records_20260207_080046.jsonl`
- **Gold answers:** `/Users/chandra/Downloads/MCQ_Evaluation_Set_Final.xlsx` (124 MCQs, column `Correct_Answer`)
