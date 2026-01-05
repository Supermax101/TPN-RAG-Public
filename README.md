# TPN RAG System v3.0

**Production-Grade Clinical Question Answering for Total Parenteral Nutrition**

A Retrieval-Augmented Generation (RAG) system for answering clinical TPN questions based on ASPEN guidelines, NICU protocols, and pediatric nutrition handbooks. Includes **Citation Grounding** to fix hallucinated citations from fine-tuned models.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [System Architecture](#system-architecture)
3. [Complete Data Flow](#complete-data-flow)
4. [Project Structure](#project-structure)
5. [Module Reference](#module-reference)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Pipeline Details](#pipeline-details)
   - [Ingestion Pipeline](#1-ingestion-pipeline)
   - [Retrieval Pipeline](#2-retrieval-pipeline)
   - [Generation Pipeline](#3-generation-pipeline)
   - [Citation Grounding](#4-citation-grounding-for-fine-tuned-models)
   - [Evaluation Framework](#5-evaluation-framework)
9. [Fine-Tuned Model Integration](#fine-tuned-model-integration)
10. [Configuration](#configuration)
11. [CLI Scripts](#cli-scripts)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Development](#development)
14. [Documentation](#documentation)

---

## Project Goals

**Primary Goal:** Prove that open-source models (Qwen2.5, Llama3) + RAG can match or exceed SOTA models (GPT-4, Claude) on clinical TPN questions.

**Secondary Goal:** Fix hallucinated citations from fine-tuned models by grounding them to real retrieved documents.

| Metric | Target |
|--------|--------|
| Accuracy on 941 grounded Q&A | 90%+ |
| RAG Lift over baseline | Measurable improvement |
| Open model vs SOTA gap | Closed with RAG |
| Citation accuracy (fine-tuned) | 85%+ after grounding |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TPN RAG SYSTEM ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────── INGESTION PIPELINE ─────────────────────────┐ │
│  │                                                                         │ │
│  │   76 Markdown    ┌─────────┐   ┌─────────┐   ┌─────────────────────┐   │ │
│  │   Documents  ───▶│ Cleaner │──▶│ Chunker │──▶│ Dual Indexing       │   │ │
│  │   (DPT2 OCR)     │         │   │         │   │ - ChromaDB (vector) │   │ │
│  │                  │ -21.4%  │   │ 4558    │   │ - BM25 (keyword)    │   │ │
│  │                  │ size    │   │ chunks  │   └─────────────────────┘   │ │
│  │                  └─────────┘   └─────────┘                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────── RETRIEVAL PIPELINE ─────────────────────────┐ │
│  │                                                                         │ │
│  │   User      ┌──────┐   ┌─────────────┐   ┌────────┐   ┌──────────────┐ │ │
│  │   Query ───▶│ HyDE │──▶│ Multi-Query │──▶│ Hybrid │──▶│ Cross-Encoder│ │ │
│  │             │      │   │  Expansion  │   │ Search │   │   Reranker   │ │ │
│  │             │(opt) │   │   (opt)     │   │RRF Fuse│   │   (opt)      │ │ │
│  │             └──────┘   └─────────────┘   └────────┘   └──────────────┘ │ │
│  │                                                              │          │ │
│  │                                              Top-K Relevant Chunks      │ │
│  └──────────────────────────────────────────────────────────────┼──────────┘ │
│                                                                 │            │
│                                                                 ▼            │
│  ┌─────────────────────────── GENERATION PIPELINE ────────────────────────┐ │
│  │                                                                         │ │
│  │   Retrieved    ┌─────────────────────┐   ┌───────────────────────────┐ │ │
│  │   Context  ───▶│ LLM Provider        │──▶│ Structured Response       │ │ │
│  │   + Query      │ - HuggingFace       │   │ {answer, thinking,        │ │ │
│  │                │ - OpenAI            │   │  confidence, tokens}      │ │ │
│  │                │ - Anthropic         │   └───────────────────────────┘ │ │
│  │                │ - Fine-tuned model  │              │                   │ │
│  │                └─────────────────────┘              │                   │ │
│  └─────────────────────────────────────────────────────┼───────────────────┘ │
│                                                        │                     │
│                                                        ▼                     │
│  ┌─────────────────────────── CITATION GROUNDING ─────────────────────────┐ │
│  │  (For fine-tuned models with hallucinated citations)                    │ │
│  │                                                                         │ │
│  │   Model Output     ┌──────────────────┐   ┌───────────────────────────┐│ │
│  │   with FAKE   ────▶│ Citation Grounder│──▶│ Output with VERIFIED     ││ │
│  │   citations        │ - Strip fakes    │   │ citations from RAG       ││ │
│  │   [p.999]          │ - Match to chunks│   │ [ASPEN Guidelines, p.44] ││ │
│  │                    │ - Inject real    │   └───────────────────────────┘│ │
│  │                    └──────────────────┘                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────── EVALUATION FRAMEWORK ───────────────────────┐ │
│  │                                                                         │ │
│  │   941 Q&A     ┌───────────────┐   ┌──────────────┐   ┌───────────────┐ │ │
│  │   Pairs   ───▶│ Retrieval     │──▶│ Answer       │──▶│ Statistical   │ │ │
│  │   (grounded)  │ Metrics       │   │ Metrics      │   │ Analysis      │ │ │
│  │               │ Hit@K, MRR    │   │ F1, Citation │   │ RAG Lift      │ │ │
│  │               └───────────────┘   └──────────────┘   └───────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

### Flow 1: Document Ingestion (One-time setup)

```
Raw DPT2 Markdown Files (76 documents)
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/ingestion/cleaner.py                                        │
│ - Remove OCR artifacts: page anchors, CAPTION ERROR, etc.       │
│ - Remove broken figures and navigation elements                 │
│ - Preserve clinical content: tables, dosing, procedures         │
│ - Result: 21.4% size reduction                                  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/ingestion/chunker.py                                        │
│ - Semantic chunking with 1000 char size, 200 overlap            │
│ - Clinical-aware separators (preserve tables, lists)            │
│ - Track source document and page metadata                       │
│ - Result: 4558 chunks (87% text, 13% tables)                    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/ingestion/pipeline.py                                       │
│ - Generate embeddings (sentence-transformers)                   │
│ - Store in ChromaDB with metadata                               │
│ - Build BM25 keyword index                                      │
│ - Persist to disk: ./data/chroma/ and ./data/bm25/              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
    Indexed Corpus Ready for Retrieval
    - ChromaDB: ./data/chroma/ (vector similarity)
    - BM25: ./data/bm25/ (keyword matching)
```

### Flow 2: Query Processing (Runtime)

```
User Question: "What is the protein requirement for preterm infants?"
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/retrieval/hyde.py (Optional - HyDE)                         │
│ - Generate hypothetical answer using LLM                        │
│ - "Protein requirements for preterm infants are typically       │
│    3-4 g/kg/day according to ASPEN guidelines..."               │
│ - Use this as the search query (bridges query-document gap)     │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/retrieval/multi_query.py (Optional - Query Expansion)       │
│ - Generate query variations to handle vocabulary mismatch       │
│ - Original: "protein requirement preterm infant"                │
│ - Variation 1: "amino acid needs premature baby"                │
│ - Variation 2: "neonatal protein supplementation guidelines"    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/retrieval/hybrid.py (Hybrid Search with RRF)                │
│                                                                 │
│ ┌─────────────────┐         ┌─────────────────┐                 │
│ │ Vector Search   │         │ BM25 Search     │                 │
│ │ (ChromaDB)      │         │ (Keyword)       │                 │
│ │ Semantic sim.   │         │ Term matching   │                 │
│ │ → Top 20        │         │ → Top 20        │                 │
│ └────────┬────────┘         └────────┬────────┘                 │
│          │                           │                          │
│          └───────────┬───────────────┘                          │
│                      ▼                                          │
│          ┌─────────────────────────┐                            │
│          │ RRF Fusion              │                            │
│          │ score = Σ 1/(k + rank)  │                            │
│          │ Merge & deduplicate     │                            │
│          └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/retrieval/reranker.py (Optional - Cross-Encoder)            │
│ - Model: BAAI/bge-reranker-v2-m3                                │
│ - Score each (query, chunk) pair with cross-attention           │
│ - Reorder by fine-grained relevance                             │
│ - Return top 5 most relevant chunks                             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
    Top 5 Retrieved Chunks with Metadata:
    [
      {content: "Preterm infants require 3-4 g/kg/day...",
       source: "ASPEN_Guidelines.md", page: 44, score: 0.95},
      ...
    ]
```

### Flow 3: Answer Generation

```
Retrieved Chunks + Original Question
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/models/base.py (Prompt Construction)                        │
│                                                                 │
│ System Prompt: "You are a clinical nutrition expert..."         │
│                                                                 │
│ User Prompt:                                                    │
│ "Answer the following question using the provided context.      │
│                                                                 │
│  CONTEXT:                                                       │
│  [Source: ASPEN_Guidelines.md, Page 44]                         │
│  Preterm infants require 3-4 g/kg/day of protein...             │
│                                                                 │
│  QUESTION: What is the protein requirement for preterm infants? │
│                                                                 │
│  Cite your sources using [Document Name, p.XX] format."         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/models/<provider>.py (LLM Generation)                       │
│                                                                 │
│ Providers:                                                      │
│ - huggingface_provider.py → HuggingFace Inference API / Local   │
│ - openai_provider.py      → GPT-4o, GPT-4o-mini, o1            │
│ - anthropic_provider.py   → Claude 4, Claude 3.5 Sonnet         │
│                                                                 │
│ Output: LLMResponse {                                           │
│   answer: "Protein requirements for preterm infants are...",    │
│   thinking: "Based on ASPEN guidelines...",                     │
│   tokens_used: 245,                                             │
│   latency_ms: 1200                                              │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
    Final Answer with Citations
```

### Flow 4: Citation Grounding (For Fine-Tuned Models)

```
Fine-Tuned Model Output (with HALLUCINATED citations):
"Protein requirement is 3-4 g/kg/day [Fake TPN Manual, p.999]"
                                      ↑ DOESN'T EXIST
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ app/retrieval/citation_grounding.py                             │
│                                                                 │
│ Step 1: Extract and remove hallucinated citations               │
│   Input:  "...3-4 g/kg/day [Fake TPN Manual, p.999]"           │
│   Output: "...3-4 g/kg/day"                                     │
│   Removed: 1 citation                                           │
│                                                                 │
│ Step 2: Match content to retrieved chunks                       │
│   Answer mentions: "3-4 g/kg/day", "preterm"                    │
│   Chunk content:   "require 3-4 g/kg/day of protein"            │
│   Match score: 0.92                                             │
│                                                                 │
│ Step 3: Inject verified citations                               │
│   Matched chunk: ASPEN_Guidelines.md, page 44                   │
│   Output: "...3-4 g/kg/day [ASPEN Guidelines, p.44]"           │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
    Grounded Output:
    "Protein requirement is 3-4 g/kg/day [ASPEN Guidelines, p.44]

    References:
    1. ASPEN Guidelines, page 44"
```

---

## Project Structure

```
TPN2.0RAG/
│
├── app/                              # Main application code
│   │
│   ├── ingestion/                    # Document processing pipeline
│   │   ├── __init__.py               # Exports: IngestionPipeline, DocumentCleaner, SemanticChunker
│   │   ├── cleaner.py                # Remove OCR artifacts, preserve clinical content
│   │   ├── chunker.py                # Semantic chunking with clinical-aware boundaries
│   │   └── pipeline.py               # Orchestrate full ingestion workflow
│   │
│   ├── retrieval/                    # Advanced retrieval techniques
│   │   ├── __init__.py               # Exports: RetrievalPipeline, HybridRetriever, CitationGrounder
│   │   ├── hybrid.py                 # Vector + BM25 with Reciprocal Rank Fusion
│   │   ├── hyde.py                   # Hypothetical Document Embeddings
│   │   ├── multi_query.py            # Query expansion for vocabulary mismatch
│   │   ├── reranker.py               # Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
│   │   ├── pipeline.py               # Unified retrieval pipeline with config
│   │   └── citation_grounding.py     # Fix hallucinated citations from fine-tuned models
│   │
│   ├── models/                       # LLM provider abstraction
│   │   ├── __init__.py               # Exports: create_model, search_hf_models, LLMProvider
│   │   ├── base.py                   # LLMProvider protocol, prompts, LLMResponse dataclass
│   │   ├── huggingface_provider.py   # HuggingFace Hub API + local inference
│   │   ├── openai_provider.py        # OpenAI GPT-4o, o1
│   │   └── anthropic_provider.py     # Anthropic Claude 4, Claude 3.5
│   │
│   ├── evaluation/                   # Testing and metrics
│   │   ├── __init__.py               # Exports: EvaluationHarness, ModelComparison, CitationEvaluator
│   │   ├── dataset.py                # Load 941 grounded Q&A pairs from JSONL
│   │   ├── metrics.py                # Retrieval metrics (Hit@K, MRR) + Answer metrics (F1)
│   │   ├── harness.py                # Evaluation orchestration
│   │   ├── comparison.py             # Multi-model RAG vs No-RAG comparison
│   │   └── citation_metrics.py       # Citation quality metrics (faithfulness, accuracy)
│   │
│   └── __init__.py                   # Lazy imports to avoid LangChain chain loading
│
├── scripts/                          # CLI tools
│   ├── ingest.py                     # Run document ingestion
│   ├── retrieve.py                   # Test retrieval pipeline
│   ├── evaluate.py                   # Run evaluation on Q&A dataset
│   ├── compare_models.py             # Compare multiple models with/without RAG
│   └── finetune_with_rag.py          # Demo: Fine-tuned model + RAG + Citation Grounding
│
├── docs/                             # Documentation
│   └── CITATION_GROUNDING.md         # Detailed explanation of citation hallucination fix
│
├── data/                             # Data directory
│   ├── documents/                    # 76 DPT2 markdown files (input)
│   ├── chroma/                       # ChromaDB vector store (generated)
│   └── bm25/                         # BM25 keyword index (generated)
│
├── eval/                             # Evaluation datasets and results
│   ├── rag_evaluation.py             # Legacy evaluation
│   └── rag_metrics.py                # Legacy metrics
│
├── pyproject.toml                    # Python 3.11+ dependencies
├── CLAUDE.md                         # Claude Code quick reference
├── TECHNICAL_SPEC.md                 # Full implementation specification
├── SENIOR_AI_ANALYSIS.md             # Root cause analysis of original issues
└── REFACTORING_PLAN.md               # Original roadmap
```

---

## Module Reference

### Ingestion Module (`app/ingestion/`)

| File | Class/Function | Description |
|------|----------------|-------------|
| `cleaner.py` | `DocumentCleaner` | Removes OCR artifacts while preserving clinical content |
| `cleaner.py` | `CleaningStats` | Dataclass with cleaning statistics |
| `chunker.py` | `SemanticChunker` | Chunks documents with clinical-aware boundaries |
| `chunker.py` | `Chunk` | Dataclass representing a document chunk |
| `pipeline.py` | `IngestionPipeline` | Orchestrates full ingestion workflow |
| `pipeline.py` | `IngestionStats` | Dataclass with ingestion statistics |

### Retrieval Module (`app/retrieval/`)

| File | Class/Function | Description |
|------|----------------|-------------|
| `hybrid.py` | `HybridRetriever` | Vector + BM25 search with RRF fusion |
| `hybrid.py` | `RRFConfig` | Configuration for RRF parameters |
| `hyde.py` | `HyDERetriever` | Hypothetical Document Embeddings |
| `hyde.py` | `HyDEConfig` | HyDE configuration |
| `multi_query.py` | `MultiQueryRetriever` | Query expansion for vocabulary mismatch |
| `reranker.py` | `CrossEncoderReranker` | Cross-encoder reranking |
| `pipeline.py` | `RetrievalPipeline` | Unified pipeline combining all techniques |
| `pipeline.py` | `RetrievalConfig` | Enable/disable each component |
| `citation_grounding.py` | `CitationGrounder` | Fix hallucinated citations |
| `citation_grounding.py` | `GroundingResult` | Result with grounded text |

### Models Module (`app/models/`)

| File | Class/Function | Description |
|------|----------------|-------------|
| `base.py` | `LLMProvider` | Abstract base class for all providers |
| `base.py` | `LLMResponse` | Dataclass with answer, thinking, tokens |
| `base.py` | `ModelConfig` | Temperature, max_tokens, etc. |
| `huggingface_provider.py` | `HuggingFaceProvider` | HuggingFace Inference API + local |
| `huggingface_provider.py` | `search_models()` | Search HuggingFace Hub dynamically |
| `huggingface_provider.py` | `list_trending_models()` | Get popular models |
| `openai_provider.py` | `OpenAIProvider` | OpenAI GPT-4o, o1 |
| `anthropic_provider.py` | `AnthropicProvider` | Anthropic Claude models |
| `__init__.py` | `create_model()` | Factory function for any provider |

### Evaluation Module (`app/evaluation/`)

| File | Class/Function | Description |
|------|----------------|-------------|
| `dataset.py` | `EvaluationDataset` | Load 941 Q&A pairs from JSONL |
| `dataset.py` | `QAPair` | Question, answer, source, page |
| `metrics.py` | `RetrievalMetrics` | Hit@K, MRR, source matching |
| `metrics.py` | `AnswerMetrics` | F1, exact match, key phrase overlap |
| `harness.py` | `EvaluationHarness` | Run full evaluation pipeline |
| `comparison.py` | `ModelComparison` | Compare RAG vs No-RAG |
| `comparison.py` | `ComparisonResult` | Results with RAG lift |
| `comparison.py` | `statistical_significance()` | t-test, Wilcoxon, Cohen's d |
| `citation_metrics.py` | `CitationEvaluator` | Measure citation quality |
| `citation_metrics.py` | `CitationResult` | Source accuracy, faithfulness |

---

## Installation

### Prerequisites

- Python 3.11+
- HuggingFace account (for model access)

### Install

```bash
# Clone repository
git clone <repo>
cd TPN2.0RAG

# Install with pip
pip install -e .

# Or with uv (faster)
uv sync

# Set API keys
export HF_TOKEN=your_huggingface_token
export OPENAI_API_KEY=your_openai_key      # Optional
export ANTHROPIC_API_KEY=your_anthropic_key # Optional
```

---

## Quick Start

### 1. Ingest Documents

```bash
# One-time setup: Process documents and create indexes
python scripts/ingest.py --docs-dir data/documents --persist-dir ./data
```

### 2. Test Retrieval

```bash
# Demo mode
python scripts/retrieve.py --demo

# Query your indexed documents
python scripts/retrieve.py --persist-dir ./data \
    --query "What is the protein requirement for preterm infants?"
```

### 3. Compare Models

```bash
# List available models (fetches from HuggingFace Hub)
python scripts/compare_models.py --list-models

# Compare HuggingFace model vs OpenAI
python scripts/compare_models.py \
    --persist-dir ./data \
    --models hf:Qwen/Qwen2.5-7B-Instruct openai:gpt-4o-mini \
    -n 50
```

### 4. Run Evaluation

```bash
# Evaluate on 100 Q&A pairs
python scripts/evaluate.py --persist-dir ./data -n 100
```

### 5. Fix Hallucinated Citations (Fine-Tuned Model)

```bash
# Demo showing how citation grounding works
python scripts/finetune_with_rag.py --demo

# With your fine-tuned model
python scripts/finetune_with_rag.py \
    --model hf:your-finetuned-model \
    --persist-dir ./data \
    --query "What is the sodium requirement for VLBW infants?"
```

---

## Pipeline Details

### 1. Ingestion Pipeline

**Purpose:** Transform raw OCR output into clean, indexed documents.

```python
from app.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    docs_dir="data/documents",
    persist_dir="./data",
    chunk_size=1000,
    chunk_overlap=200,
)
stats = pipeline.run()

print(f"Documents processed: {stats.total_documents}")
print(f"Chunks created: {stats.total_chunks}")
print(f"Size reduction: {stats.cleaning_reduction:.1%}")
```

**Cleaning Operations:**
- Remove page anchors (`<!-- PageNumber: "1" -->`)
- Remove OCR errors (`CAPTION ERROR`, broken figures)
- Remove navigation elements
- Preserve clinical tables and dosing information

**Chunking Strategy:**
- 1000 character chunks with 200 character overlap
- Clinical-aware separators (don't split tables)
- Preserve paragraph boundaries
- Track source document and page metadata

### 2. Retrieval Pipeline

**Purpose:** Find the most relevant documents for a query.

```python
from app.retrieval import RetrievalPipeline, RetrievalConfig

config = RetrievalConfig(
    enable_hyde=True,         # Hypothetical Document Embeddings
    enable_multi_query=True,  # Query expansion
    enable_reranking=True,    # Cross-encoder reranking
    vector_weight=0.5,        # Weight for vector search
    bm25_weight=0.5,          # Weight for BM25 search
    initial_k=20,             # Candidates before reranking
    final_top_k=5,            # Final results
)

pipeline = RetrievalPipeline.from_persisted("./data", config=config)
results = pipeline.retrieve("protein requirements for preterm infants")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.metadata['source']}, Page: {result.metadata['page']}")
    print(f"Content: {result.content[:200]}...")
```

**Retrieval Techniques:**

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **Hybrid Search** | Combine semantic + keyword | Vector (ChromaDB) + BM25 with RRF |
| **HyDE** | Bridge query-document gap | Generate hypothetical answer as query |
| **Multi-Query** | Handle vocabulary mismatch | Expand query with variations |
| **Cross-Encoder** | Fine-grained relevance | BAAI/bge-reranker-v2-m3 |

### 3. Generation Pipeline

**Purpose:** Generate answers using LLMs with retrieved context.

```python
from app.models import create_model, search_hf_models

# Discover models dynamically
models = search_hf_models("Qwen instruct", limit=10)
print(f"Found: {[m['id'] for m in models]}")

# Create provider
model = create_model("hf", "Qwen/Qwen2.5-7B-Instruct")

# Generate with RAG
response = model.generate(
    question="What is the protein requirement for preterm infants?",
    context=retrieved_context,  # From retrieval pipeline
    use_rag=True,
)

print(f"Answer: {response.answer}")
print(f"Tokens: {response.tokens_used}")
print(f"Latency: {response.latency_ms:.0f}ms")
```

**Supported Providers:**

| Provider | Models | Usage |
|----------|--------|-------|
| **HuggingFace** | Qwen, Llama, Mistral, Gemma, any Hub model | `create_model("hf", "Qwen/Qwen2.5-7B-Instruct")` |
| **OpenAI** | GPT-4o, GPT-4o-mini, o1-preview | `create_model("openai", "gpt-4o")` |
| **Anthropic** | Claude 4, Claude 3.5 Sonnet | `create_model("anthropic", "claude-sonnet-4-20250514")` |

### 4. Citation Grounding (For Fine-Tuned Models)

**Purpose:** Fix hallucinated citations from fine-tuned models.

**The Problem:**
```
Fine-tuned model output:
"Protein is 3-4 g/kg/day [Fake TPN Manual, p.999]"
                         ↑ HALLUCINATED (doesn't exist)
```

**The Solution:**
```python
from app.retrieval import CitationGrounder

grounder = CitationGrounder()

# Fine-tuned model output (correct answer, fake citations)
model_output = """
Protein requirements for preterm infants are 3-4 g/kg/day
[TPN Nutrition Manual 2024, p.234]. This should be initiated
within the first 24 hours [Clinical Handbook, p.567].
"""

# Real chunks from RAG retrieval
retrieved_chunks = [
    {
        "content": "Preterm infants require 3-4 g/kg/day of protein...",
        "metadata": {"source": "ASPEN_Guidelines.md", "page": 44}
    }
]

# Ground citations to reality
result = grounder.ground_citations(
    model_output,
    retrieved_chunks,
    add_inline_citations=True,
    add_references_section=True,
)

print(result.grounded_text)
# Output:
# "Protein requirements for preterm infants are 3-4 g/kg/day
#  [ASPEN Guidelines, p.44]. This should be initiated within
#  the first 24 hours [ASPEN Guidelines, p.44].
#
#  References:
#  1. ASPEN Guidelines, page 44"

print(f"Citations removed: {result.citations_removed}")
print(f"Citations added: {result.citations_added}")
print(f"Confidence: {result.confidence:.1%}")
```

### 5. Evaluation Framework

**Purpose:** Measure system quality and compare approaches.

```python
from app.evaluation import ModelComparison, CitationEvaluator

# Compare multiple models
comparison = ModelComparison(
    dataset_path="/path/to/test.jsonl",
    retriever=retriever,
    top_k=5,
)

comparison.add_model("hf", "Qwen/Qwen2.5-7B-Instruct")
comparison.add_model("openai", "gpt-4o-mini")

results = comparison.run(
    sample_size=100,
    include_baseline=True,  # Also test without RAG
    save_results=True,
)

print(results.to_markdown())

# Get RAG lift for each model
for model in results.models:
    lift = results.get_rag_lift(model.model_name)
    print(f"{model.model_name}: {lift:+.1%} improvement with RAG")
```

**Citation Quality Evaluation:**
```python
from app.evaluation import CitationEvaluator, RetrievedChunk

evaluator = CitationEvaluator()

result = evaluator.evaluate(
    question="What is the protein requirement?",
    generated_answer=grounded_answer,
    retrieved_chunks=chunks,
    ground_truth_source="ASPEN Guidelines",
    ground_truth_page=44,
)

print(f"Source Accuracy: {result.source_accuracy:.1%}")
print(f"Page Precision: {result.page_precision:.1%}")
print(f"Faithfulness: {result.faithfulness_score:.1%}")
print(f"Hallucination Risk: {result.hallucination_risk:.1%}")
```

---

## Fine-Tuned Model Integration

If you have a fine-tuned TPN model that hallucinates citations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINE-TUNED MODEL + RAG PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User asks question                                                      │
│     "What is the sodium requirement for VLBW infants?"                      │
│                                                                             │
│  2. RAG retrieves real documents                                            │
│     → ASPEN_Guidelines.md, page 52: "Sodium 3-5 mEq/kg/day..."             │
│                                                                             │
│  3. Fine-tuned model generates answer (with RAG context)                    │
│     → "Sodium is 3-5 mEq/kg/day [Fake Manual, p.999]"                      │
│                                    ↑ HALLUCINATED                           │
│                                                                             │
│  4. Citation Grounding fixes citations                                      │
│     → "Sodium is 3-5 mEq/kg/day [ASPEN Guidelines, p.52]"                  │
│                                    ↑ VERIFIED                               │
│                                                                             │
│  Result: Correct answer + Verifiable citations                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this works:**
- Fine-tuned model learned clinical facts correctly
- Fine-tuned model learned citation FORMAT (but fabricates sources)
- RAG provides REAL documents with actual page numbers
- Citation Grounding replaces fake citations with verified ones

**No retraining required!**

---

## Configuration

### Retrieval Configuration

```python
from app.retrieval import RetrievalConfig

config = RetrievalConfig(
    # Enable/disable components
    enable_hyde=True,
    enable_multi_query=True,
    enable_reranking=True,

    # Search weights
    vector_weight=0.5,
    bm25_weight=0.5,

    # Result counts
    initial_k=20,      # Candidates before reranking
    final_top_k=5,     # Final results

    # Models
    reranker_model="BAAI/bge-reranker-v2-m3",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
```

### Model Configuration

```python
from app.models import ModelConfig

config = ModelConfig(
    temperature=0.0,      # Deterministic for clinical use
    max_tokens=1024,
    top_p=1.0,
    include_thinking=True,
)
```

### Chunking Configuration

```python
from app.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    docs_dir="data/documents",
    persist_dir="./data",
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
)
```

---

## CLI Scripts

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/ingest.py` | Process and index documents | `python scripts/ingest.py --docs-dir data/documents` |
| `scripts/retrieve.py` | Test retrieval pipeline | `python scripts/retrieve.py --query "protein requirements"` |
| `scripts/evaluate.py` | Run evaluation on dataset | `python scripts/evaluate.py -n 100` |
| `scripts/compare_models.py` | Compare multiple models | `python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct` |
| `scripts/finetune_with_rag.py` | Demo citation grounding | `python scripts/finetune_with_rag.py --demo` |

---

## Evaluation Metrics

### Retrieval Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hit@1** | Correct source in top 1 result | >70% |
| **Hit@3** | Correct source in top 3 results | >85% |
| **Hit@5** | Correct source in top 5 results | >90% |
| **MRR** | Mean Reciprocal Rank | >0.75 |

### Answer Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **F1 Score** | Token overlap with reference | >80% |
| **Exact Match** | Exact answer match | >50% |
| **Key Phrase Overlap** | Clinical terms preserved | >85% |

### Citation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Source Accuracy** | Cited correct document | >90% |
| **Page Precision** | Cited correct page | >80% |
| **Faithfulness** | Answer matches cited context | >85% |
| **Hallucination Risk** | Unsupported claims | <15% |

### RAG Lift

```
RAG Lift = (RAG_F1 - Baseline_F1) / Baseline_F1

Example: If baseline F1 is 60% and RAG F1 is 75%
RAG Lift = (0.75 - 0.60) / 0.60 = +25%
```

---

## Development

```bash
# Format code
black app scripts

# Lint
ruff check app scripts

# Type check
mypy app

# Run tests
pytest
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | This file - comprehensive overview |
| [CLAUDE.md](./CLAUDE.md) | Quick reference for Claude Code |
| [TECHNICAL_SPEC.md](./TECHNICAL_SPEC.md) | Full implementation specification |
| [docs/CITATION_GROUNDING.md](./docs/CITATION_GROUNDING.md) | Citation hallucination fix details |
| [SENIOR_AI_ANALYSIS.md](./SENIOR_AI_ANALYSIS.md) | Root cause analysis |
| [REFACTORING_PLAN.md](./REFACTORING_PLAN.md) | Original roadmap |

---

## Disclaimer

This system is for research and educational purposes. Clinical decisions should be validated by qualified healthcare professionals. Current accuracy requires further validation before production clinical use.

---

## License

[Specify license]
