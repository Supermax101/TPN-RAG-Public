# TPN-RAG: Clinical Nutrition RAG Benchmark System

A retrieval-augmented generation (RAG) system for Total Parenteral Nutrition (TPN) clinical decision support, with a publishable benchmark framework evaluating 5 SOTA LLMs across 5 prompt strategies.

## What This Project Does

1. **Knowledge Base** -- Ingests clinical TPN documents (ASPEN guidelines, NICU protocols, nutrition handbooks) into a hybrid retrieval index (ChromaDB vectors + BM25 keywords)
2. **RAG Pipeline** -- Retrieves relevant context for clinical questions using iterative query decomposition, hybrid search, reciprocal rank fusion, and cross-encoder reranking
3. **Benchmark Framework** -- Evaluates LLM accuracy on TPN MCQ and open-ended questions under controlled conditions (shared retrieval context for fairness, parallel async execution, full run ledger)
4. **5 SOTA Models** -- OpenAI GPT-4o, Anthropic Claude Sonnet 4.5, Google Gemini 2.5 Pro, xAI Grok-4, Moonshot Kimi K2
5. **5 Prompt Strategies** -- Zero-Shot, Few-Shot, Chain-of-Thought, CoT-SC (majority vote), Retrieval-Augmented Prompting

## Project Structure

```
app/
  config.py              # Settings from env vars / .env file
  providers/             # LLM provider implementations (async)
    base.py              # LLMProvider ABC + generate_structured()
    openai.py            # GPT-4o, GPT-5, O1/O3 reasoning models
    anthropic.py         # Claude (tool_use structured output)
    gemini.py            # Gemini 2.5 (response_schema JSON)
    xai.py               # Grok-4 (OpenAI-compatible)
    kimi.py              # Kimi K2 (OpenAI-compatible + rate limiting)
    huggingface.py       # Sync provider for HF models
  evaluation/            # Benchmark framework
    benchmark_runner.py  # Main orchestrator (async parallel)
    benchmark_types.py   # Pydantic schemas (ExperimentConfig, RunRecord, etc.)
    provider_adapter.py  # Unified async wrapper + generate_structured()
    retriever_adapter.py # Deterministic retrieval with fairness snapshots
    agentic_retriever.py # LLM relevance judge + query rewrite (Self-RAG/CRAG)
    metrics.py           # Token F1, exact match, key-phrase overlap, citation match
    statistics.py        # Cohen's kappa, Fleiss, McNemar, bootstrap CI
    benchmark_analysis.py
    citation_metrics.py
    data_leakage.py
  prompting/             # Prompt templates and rendering
    renderer.py          # Template-backed renderer with dynamic few-shot
    example_pool.py      # Embedding-indexed FewShotPool (cosine similarity)
    example_data.py      # 16 curated TPN MCQ examples
    templates/           # ZS, FEW_SHOT, COT, COT_SC, RAP .txt files
  retrieval/             # Advanced retrieval stack
    hybrid.py            # Dense + BM25 with RRF
    hyde.py              # Hypothetical Document Embeddings
    multi_query.py       # Query expansion
    reranker.py          # Cross-encoder reranking
    pipeline.py          # Unified RetrievalPipeline
    citation_grounding.py # Fix hallucinated citations (+ NLI blending)
    nli_grounding.py     # NLI entailment verification
    tokenizer.py         # Clinical-aware BM25 tokenizer
  ingestion/             # Document processing
    pipeline.py          # Full ingestion orchestrator
    chunker.py           # Recursive clinical-aware chunker
    semantic_chunker.py  # Embedding-boundary chunker (opt-in)
    cleaner.py           # OCR artifact removal
  parsers/
    mcq_parser.py        # 5-strategy regex parser + Pydantic MCQAnswer
  chains/
    agentic_rag.py       # LangGraph agentic RAG (reference implementation)
    tpn_prompts.py       # Clinical prompt templates

scripts/
  tpnctl.py              # CLI: status, ingest, benchmark, list-models, quick-test, ask
  run_benchmark.py       # Direct benchmark runner script
  ask_question.py        # Single question helper
  vast_bootstrap.sh      # Set up a fresh vast.ai instance
  vast_run_benchmark.sh  # End-to-end benchmark on vast.ai

tests/
  test_metrics.py
  test_agentic_retriever.py
  test_structured_output.py
  test_benchmark_runner.py
  test_templates_and_parser.py
  test_statistics.py
  test_config_and_prompting.py
  test_data_leakage.py
  test_benchmark_analysis.py

data/
  documents/             # 70+ clinical TPN markdown files (knowledge base)
  eval/                  # Excel evaluation sets (MCQ + open-ended)

eval/
  data/                  # JSONL holdout datasets (generated from Excel)
  results/               # Benchmark outputs (gitignored, regenerated each run)
```

## SOTA Features (All Opt-In)

| Feature | Config Flag | Default | What It Does |
|---------|------------|---------|--------------|
| Agentic retrieval | `agentic_retrieval=True` | OFF | LLM judges chunk relevance, rewrites query if <50% relevant |
| Structured output | Automatic rescue | OFF (rescue-only) | JSON-mode parsing fires only when regex returns PARSE_ERROR |
| Dynamic few-shot | `dynamic_few_shot=True` | OFF | Selects 2 most similar examples from 16-example pool via cosine similarity |
| NLI grounding | `use_nli=True` on CitationGrounder | OFF | Cross-encoder entailment blended 50/50 with heuristic; contradiction forces score to 0 |
| Semantic chunking | `chunker_type="semantic"` or `CHUNKER_TYPE=semantic` | OFF | Splits at topic boundaries via embedding similarity drops |

All features default to OFF. Existing benchmark results are fully reproducible with identical outputs.

## Requirements

- Python 3.11+
- API keys for LLM providers (set in `.env`)
- ~8GB RAM for embedding models (sentence-transformers)
- GPU recommended for cross-encoder reranking and NLI

## What Needs To Be Done Now

### 1. Run Tests on vast.ai

The project requires Python 3.11+ and heavy ML dependencies (sentence-transformers, chromadb, langgraph) that need a proper compute environment. Tests should run on vast.ai.

**Steps:**

```bash
# On vast.ai instance:

# 1. Clone the repo
git clone https://github.com/Supermax101/TPN-RAG-Public.git
cd TPN-RAG-Public

# 2. Bootstrap environment
bash scripts/vast_bootstrap.sh https://github.com/Supermax101/TPN-RAG-Public.git main

# 3. Activate venv
cd ~/work/tpn-rag/TPN-RAG-Public
source .venv/bin/activate

# 4. Run unit tests
pytest tests/ -v

# 5. Run specific test suites
pytest tests/test_metrics.py -v              # Open-ended eval metrics
pytest tests/test_structured_output.py -v    # JSON mode + fallback
pytest tests/test_agentic_retriever.py -v    # Mock LLM judge
pytest tests/test_templates_and_parser.py -v # Prompt rendering + MCQ parsing
pytest tests/test_benchmark_runner.py -v     # Full benchmark runner
pytest tests/test_statistics.py -v           # Statistical tests
```

### 2. Run Full Benchmark (After Tests Pass)

```bash
# Fill in API keys
nano .env

# Convert Excel evaluation sets to JSONL
python scripts/convert_eval_xlsx.py --out-dir eval/data/benchmark_2026-02-05

# Build retrieval indexes (requires OPENAI_API_KEY for embeddings)
python scripts/tpnctl.py ingest --docs-dir data/documents --persist-dir ./data

# Run benchmark (5 models x 5 strategies x 5 repeats)
python scripts/tpnctl.py benchmark \
  --mcq-dataset eval/data/benchmark_2026-02-05/mcq_holdout.jsonl \
  --persist-dir ./data \
  --models gpt-4o,claude-sonnet,gemini-2.5-pro,grok-4,kimi-k2

# Or run everything end-to-end:
bash scripts/vast_run_benchmark.sh
```

### 3. Test SOTA Features (After Baseline Benchmark)

```bash
# Test agentic retrieval
python scripts/tpnctl.py benchmark \
  --mcq-dataset eval/data/benchmark_2026-02-05/mcq_holdout.jsonl \
  --persist-dir ./data \
  --models gpt-4o \
  --agentic-retrieval

# Test dynamic few-shot
python scripts/tpnctl.py benchmark \
  --mcq-dataset eval/data/benchmark_2026-02-05/mcq_holdout.jsonl \
  --persist-dir ./data \
  --models gpt-4o \
  --dynamic-few-shot

# Quick smoke test
python scripts/tpnctl.py quick-test --provider openai --with-rag
python scripts/tpnctl.py ask "What is the protein requirement for a 28-week preterm infant?" \
  --provider openai --strategy FEW_SHOT --with-rag
```

### 4. Re-Ingest After BM25 Fix

The BM25 tokenizer was fixed to use `clinical_tokenize()` instead of `.lower().split()`. After the fix, the BM25 index must be rebuilt:

```bash
python scripts/tpnctl.py ingest --docs-dir data/documents --persist-dir ./data
# Verify: inspect data/bm25/tokenized.json for clinical tokens like "g_per_kg_per_day"
```

## API Keys Required

| Provider | Env Variable | Used For |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | GPT-4o generation + text-embedding-3-large |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Sonnet 4.5 generation |
| Google | `GEMINI_API_KEY` | Gemini 2.5 Pro/Flash generation |
| xAI | `XAI_API_KEY` | Grok-4 generation |
| Moonshot | `KIMI_API_KEY` | Kimi K2 generation |

## Reproducibility

- All SOTA features default to OFF -- toggling them on is explicit via config flags
- Structured output is rescue-only (fires only on regex PARSE_ERROR)
- Dynamic few-shot uses deterministic cosine similarity (no randomness)
- All LLM judge calls use `temperature=0.0`
- Seed (`42`) is set at benchmark start and propagated
- Retrieval snapshots are shared across models for fair comparison
- Full run ledger (JSONL) captures every generation for audit
