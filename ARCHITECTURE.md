# TPN RAG System - Complete Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TPN RAG SYSTEM v2.1.0                                 │
│                    Clinical Q&A for Total Parenteral Nutrition                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Goal: Prove open models (Qwen2.5, Llama3) + RAG >= SOTA (GPT-4, Claude)       │
│  Target: 90%+ accuracy on 941 grounded Q&A pairs                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA LAYER                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  76 MD Source   │    │  4,558 Chunks   │    │  941 Q&A Pairs (test.jsonl) │  │
│  │  Documents      │    │  (ChromaDB)     │    │  with source citations      │  │
│  │  data/documents/│    │  data/chromadb/ │    │                             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                │                      │                        │
                ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                                   │
│                              app/ingestion/                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────────────┐  │
│  │ DocumentLoader│───►│ DocumentCleaner  │───►│    SemanticChunker           │  │
│  │              │    │ cleaner.py       │    │    chunker.py                 │  │
│  │ Load MD/PDF  │    │ • Remove OCR     │    │ • chunk_size=1000             │  │
│  │              │    │   artifacts      │    │ • overlap=200                 │  │
│  │              │    │ • Clean DPT2     │    │ • Clinical-aware splits       │  │
│  │              │    │ • 21.4% reduction│    │ • Table detection             │  │
│  └──────────────┘    └──────────────────┘    └───────────────────────────────┘  │
│                                                             │                     │
│                                                             ▼                     │
│                                              ┌───────────────────────────────┐   │
│                                              │    IngestionPipeline          │   │
│                                              │    pipeline.py                │   │
│                                              │ • Orchestrates full workflow  │   │
│                                              │ • Builds ChromaDB + BM25      │   │
│                                              │ • Persists to data/           │   │
│                                              └───────────────────────────────┘   │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL PIPELINE                                   │
│                              app/retrieval/                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  User Query: "What is the protein requirement for preterm infants?"              │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                        QUERY EXPANSION (Optional)                         │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌─────────────────────┐       ┌─────────────────────────────────────┐   │    │
│  │  │  HyDERetriever      │       │  MultiQueryRetriever                │   │    │
│  │  │  hyde.py            │       │  multi_query.py                     │   │    │
│  │  │                     │       │                                     │   │    │
│  │  │  Generate hypothetic│       │  Expand query into 3-5 variants:    │   │    │
│  │  │  answer, then embed │       │  • Original query                   │   │    │
│  │  │  that for search    │       │  • Synonym variations               │   │    │
│  │  │                     │       │  • Clinical rephrasing              │   │    │
│  │  └─────────────────────┘       └─────────────────────────────────────┘   │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         HYBRID SEARCH                                     │    │
│  │                         hybrid.py                                         │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌───────────────────────┐         ┌───────────────────────────────┐     │    │
│  │  │   Vector Search       │         │    BM25 Search                │     │    │
│  │  │   (ChromaDB)          │         │    (Sparse Index)             │     │    │
│  │  │                       │         │                               │     │    │
│  │  │   Embedding Models:   │         │    Keyword matching           │     │    │
│  │  │   • Qwen3-Embedding-8B│         │    Exact term recall          │     │    │
│  │  │   • MedEmbed-large    │         │                               │     │    │
│  │  │   • KaLM-Gemma3-12B   │         │                               │     │    │
│  │  │                       │         │                               │     │    │
│  │  │   weight: 0.5         │         │    weight: 0.5                │     │    │
│  │  └───────────────────────┘         └───────────────────────────────┘     │    │
│  │              │                                 │                          │    │
│  │              └─────────────┬───────────────────┘                          │    │
│  │                            ▼                                              │    │
│  │               ┌───────────────────────────┐                               │    │
│  │               │  Reciprocal Rank Fusion   │                               │    │
│  │               │  (RRF, k=60)              │                               │    │
│  │               │                           │                               │    │
│  │               │  score = Σ 1/(k + rank_i) │                               │    │
│  │               └───────────────────────────┘                               │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         RERANKING                                         │    │
│  │                         reranker.py                                       │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │    │
│  │  │  CrossEncoderReranker                                           │     │    │
│  │  │  Model: BAAI/bge-reranker-v2-m3                                 │     │    │
│  │  │                                                                 │     │    │
│  │  │  • Takes query + each candidate                                 │     │    │
│  │  │  • Computes relevance score                                     │     │    │
│  │  │  • Reorders by cross-attention score                            │     │    │
│  │  │  • Returns top_k most relevant                                  │     │    │
│  │  └─────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │  RetrievalPipeline (pipeline.py) - Unified Interface                     │    │
│  │  • Configurable: enable_hyde, enable_multi_query, enable_reranking       │    │
│  │  • Returns: List[RetrievedDocument] with scores and metadata             │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              GENERATION LAYER                                     │
│                              app/models/ + app/services/                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Retrieved Context + Question                                                     │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         MODEL PROVIDERS                                   │    │
│  │                         app/models/                                       │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │    │
│  │  │ HuggingFace     │  │ OpenAI          │  │ Anthropic               │   │    │
│  │  │ Provider        │  │ Provider        │  │ Provider                │   │    │
│  │  │                 │  │                 │  │                         │   │    │
│  │  │ • Qwen2.5-7B    │  │ • gpt-4o        │  │ • claude-sonnet-4       │   │    │
│  │  │ • Llama-3.1-8B  │  │ • gpt-4o-mini   │  │ • claude-3.5-sonnet     │   │    │
│  │  │ • tpn-gpt-oss   │  │ • o1-preview    │  │ • claude-3-opus         │   │    │
│  │  │   -20b (custom) │  │                 │  │                         │   │    │
│  │  │                 │  │                 │  │                         │   │    │
│  │  │ Uses:           │  │ Uses:           │  │ Uses:                   │   │    │
│  │  │ HF Inference API│  │ OpenAI API      │  │ Anthropic API           │   │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘   │    │
│  │           │                   │                        │                  │    │
│  │           └───────────────────┼────────────────────────┘                  │    │
│  │                               ▼                                           │    │
│  │              ┌────────────────────────────────────┐                       │    │
│  │              │  LLMProvider (base.py)             │                       │    │
│  │              │  Unified Interface:                │                       │    │
│  │              │  • generate(question, context)     │                       │    │
│  │              │  • Returns: LLMResponse            │                       │    │
│  │              └────────────────────────────────────┘                       │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│       │                                                                           │
│       ▼                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         RAG SERVICE                                       │    │
│  │                         app/services/rag.py                               │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │    │
│  │  │  RAGService                                                     │     │    │
│  │  │  • search(query) → SearchResponse                               │     │    │
│  │  │  • ask(question) → RAGResponse                                  │     │    │
│  │  │                                                                 │     │    │
│  │  │  Combines:                                                      │     │    │
│  │  │  • EmbeddingProvider                                            │     │    │
│  │  │  • VectorStore                                                  │     │    │
│  │  │  • LLMProvider                                                  │     │    │
│  │  └─────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │    │
│  │  │  CitationGrounder (app/retrieval/citation_grounding.py)         │     │    │
│  │  │  • For fine-tuned models that may hallucinate citations         │     │    │
│  │  │  • Grounds citations to actual retrieved chunks                 │     │    │
│  │  └─────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION FRAMEWORK                                 │
│                              app/evaluation/ + eval/                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         DATASET                                           │    │
│  │                         dataset.py                                        │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │    │
│  │  │  EvaluationDataset                                              │     │    │
│  │  │  • Loads 941 Q&A pairs from test.jsonl                          │     │    │
│  │  │  • Each pair has: question, answer, source_doc, page_num        │     │    │
│  │  │  • Grounded to specific document locations                      │     │    │
│  │  └─────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         METRICS                                           │    │
│  │                         metrics.py                                        │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────────┐  │    │
│  │  │  RetrievalMetrics       │    │  AnswerMetrics                      │  │    │
│  │  │                         │    │                                     │  │    │
│  │  │  • Hit@K (K=1,3,5,10)   │    │  • F1 Score (token overlap)         │  │    │
│  │  │  • MRR (Mean Recip Rank)│    │  • Exact Match                      │  │    │
│  │  │  • Source Match Rate    │    │  • Key Phrase Overlap               │  │    │
│  │  │  • Page Accuracy        │    │  • Semantic Similarity              │  │    │
│  │  └─────────────────────────┘    └─────────────────────────────────────┘  │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                         EVALUATION SCRIPTS                                │    │
│  │                         eval/                                             │    │
│  ├──────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                           │    │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐ │    │
│  │  │ clinical_eval.py  │  │ batch_eval.py     │  │ rag_evaluation.py     │ │    │
│  │  │                   │  │                   │  │                       │ │    │
│  │  │ Deterministic     │  │ Batch processing  │  │ Full RAG eval with    │ │    │
│  │  │ clinical value    │  │ with GPT-5-mini   │  │ multiple providers    │ │    │
│  │  │ matching          │  │ as judge          │  │                       │ │    │
│  │  │                   │  │                   │  │                       │ │    │
│  │  │ • Extracts doses  │  │ • GEval metrics   │  │ • Provider selection  │ │    │
│  │  │ • Units matching  │  │ • Correctness     │  │ • Sample size config  │ │    │
│  │  │ • Range checks    │  │ • Relevance       │  │ • Result export       │ │    │
│  │  └───────────────────┘  └───────────────────┘  └───────────────────────┘ │    │
│  │                                                                           │    │
│  │  ┌───────────────────┐  ┌───────────────────┐                            │    │
│  │  │baseline_evaluation│  │ comparison.py     │                            │    │
│  │  │.py                │  │                   │                            │    │
│  │  │                   │  │ Multi-model       │                            │    │
│  │  │ Model-only (no    │  │ comparison with   │                            │    │
│  │  │ RAG) baseline     │  │ statistical       │                            │    │
│  │  │ performance       │  │ significance      │                            │    │
│  │  │                   │  │ (t-test, Cohen's d│                            │    │
│  │  └───────────────────┘  └───────────────────┘                            │    │
│  │                                                                           │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              API / CLI LAYER                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                             FastAPI Server                                   │ │
│  │                             app/api/                                         │ │
│  ├─────────────────────────────────────────────────────────────────────────────┤ │
│  │  app.py ─── routes.py ─── dependencies.py ─── schemas.py                    │ │
│  │     │                                                                        │ │
│  │     ├── POST /ask          → RAGResponse                                     │ │
│  │     ├── POST /search       → SearchResponse                                  │ │
│  │     └── GET  /health       → HealthCheck                                     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                             CLI Scripts                                      │ │
│  │                             scripts/                                         │ │
│  ├─────────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                              │ │
│  │  scripts/ingest.py        → Build vector store from documents               │ │
│  │  scripts/retrieve.py      → Test retrieval pipeline                         │ │
│  │  scripts/evaluate.py      → Run evaluation suite                            │ │
│  │  scripts/compare_models.py→ Compare multiple models                         │ │
│  │                                                                              │ │
│  │  tpn_rag.py               → Minimal CLI (build, ask, chat)                  │ │
│  │  cli.py                   → Full CLI interface                              │ │
│  │                                                                              │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │         76 Source Documents          │
                    │         (Markdown/PDF)               │
                    │         data/documents/              │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         INGESTION PIPELINE           │
                    │                                      │
                    │  Clean → Chunk → Embed → Store       │
                    └──────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌─────────────────────┐            ┌─────────────────────┐
         │     ChromaDB        │            │     BM25 Index      │
         │   Vector Store      │            │   Sparse Index      │
         │  (4,558 chunks)     │            │                     │
         │                     │            │                     │
         │  Embedding Models:  │            │  Keyword matching   │
         │  • Qwen3-8B         │            │  for exact terms    │
         │  • MedEmbed-large   │            │                     │
         │  • KaLM-Gemma3-12B  │            │                     │
         └─────────────────────┘            └─────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │            USER QUERY                │
                    │  "What is protein dose for preterm?" │
                    └──────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌─────────────────────┐            ┌─────────────────────┐
         │   Query Expansion   │            │   Direct Search     │
         │   (Optional)        │            │                     │
         │                     │            │                     │
         │  • HyDE: Generate   │            │  Skip expansion     │
         │    hypothetical doc │            │  for simple queries │
         │  • Multi-Query:     │            │                     │
         │    3-5 variations   │            │                     │
         └─────────────────────┘            └─────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │          HYBRID SEARCH               │
                    │                                      │
                    │  Vector (0.5) + BM25 (0.5) → RRF     │
                    │  Returns: 20-50 candidates           │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         CROSS-ENCODER RERANK         │
                    │         BAAI/bge-reranker-v2-m3      │
                    │                                      │
                    │  Score each (query, doc) pair        │
                    │  Return top_k=5-10 most relevant     │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         CONTEXT ASSEMBLY             │
                    │                                      │
                    │  Format retrieved chunks with        │
                    │  source citations for prompt         │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │            LLM GENERATION            │
                    │                                      │
                    │  Model: Qwen2.5-7B / GPT-4o / Claude │
                    │                                      │
                    │  Prompt:                             │
                    │  "You are a clinical TPN expert...   │
                    │   Answer based on context only..."   │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         CITATION GROUNDING           │
                    │         (For fine-tuned models)      │
                    │                                      │
                    │  Verify citations match retrieved    │
                    │  chunks, fix hallucinated refs       │
                    └──────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │            FINAL ANSWER              │
                    │                                      │
                    │  Answer: "Protein requirement for    │
                    │  preterm infants is 3.5-4.0 g/kg/day │
                    │  per ASPEN guidelines..."            │
                    │                                      │
                    │  Sources: [ASPEN Handbook, p.42]     │
                    └──────────────────────────────────────┘
```

---

## File Structure Map

```
TPN-RAG-Public/
├── app/                              # Main application package
│   ├── __init__.py                   # Lazy imports, main exports
│   ├── config.py                     # Settings (HF_TOKEN, models, etc.)
│   ├── data_models.py                # Pydantic schemas
│   ├── rag_pipeline.py               # TPN_RAG main class
│   │
│   ├── ingestion/                    # Document processing
│   │   ├── cleaner.py                # OCR artifact removal
│   │   ├── chunker.py                # Semantic chunking
│   │   └── pipeline.py               # Full ingestion workflow
│   │
│   ├── retrieval/                    # Search & retrieval
│   │   ├── hybrid.py                 # Vector + BM25 + RRF
│   │   ├── hyde.py                   # Hypothetical doc embeddings
│   │   ├── multi_query.py            # Query expansion
│   │   ├── reranker.py               # Cross-encoder reranking
│   │   ├── citation_grounding.py     # Fix hallucinated citations
│   │   └── pipeline.py               # Unified retrieval interface
│   │
│   ├── models/                       # LLM providers
│   │   ├── base.py                   # LLMProvider protocol
│   │   ├── huggingface_provider.py   # HF models (Qwen, Llama)
│   │   ├── openai_provider.py        # GPT-4o, GPT-5-mini
│   │   └── anthropic_provider.py     # Claude models
│   │
│   ├── evaluation/                   # Metrics & evaluation
│   │   ├── dataset.py                # Load 941 Q&A pairs
│   │   ├── metrics.py                # Retrieval + Answer metrics
│   │   ├── harness.py                # Full evaluation runner
│   │   ├── comparison.py             # Multi-model comparison
│   │   └── citation_metrics.py       # Citation accuracy
│   │
│   ├── services/                     # Business logic
│   │   ├── rag.py                    # RAGService class
│   │   ├── hybrid_rag.py             # Advanced RAG
│   │   └── prompts.py                # Prompt templates
│   │
│   ├── providers/                    # External integrations
│   │   ├── embeddings.py             # HuggingFaceEmbeddingProvider
│   │   ├── vectorstore.py            # ChromaDB wrapper
│   │   ├── openai.py                 # OpenAI provider
│   │   ├── gemini.py                 # Google Gemini
│   │   └── xai.py                    # xAI Grok
│   │
│   ├── chains/                       # LangChain LCEL chains
│   │   ├── retrieval_chain.py        # Retrieval chain
│   │   ├── mcq_chain.py              # MCQ answering
│   │   └── agentic_rag.py            # Agentic RAG
│   │
│   └── api/                          # FastAPI server
│       ├── app.py                    # FastAPI app
│       ├── routes.py                 # API endpoints
│       └── dependencies.py           # DI container
│
├── eval/                             # Evaluation scripts
│   ├── clinical_eval.py              # Deterministic clinical eval
│   ├── batch_eval.py                 # Batch processing
│   ├── rag_evaluation.py             # Full RAG eval
│   ├── baseline_evaluation.py        # No-RAG baseline
│   └── ragas_evaluation.py           # RAGAS metrics
│
├── scripts/                          # CLI scripts
│   ├── ingest.py                     # Build vector store
│   ├── retrieve.py                   # Test retrieval
│   ├── evaluate.py                   # Run evaluation
│   └── compare_models.py             # Model comparison
│
├── data/                             # Data directory
│   ├── documents/                    # 76 source MD files
│   └── chromadb/                     # Persisted vector store
│
├── tpn_rag.py                        # Minimal CLI
├── cli.py                            # Full CLI
├── pyproject.toml                    # Dependencies
├── PROGRESS_TRACKER.md               # Task tracking
└── CLAUDE.md                         # AI instructions
```

---

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT DEPENDENCY GRAPH                               │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   config.py     │
                              │   (Settings)    │
                              └────────┬────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
            ▼                          ▼                          ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │   ingestion/    │       │   retrieval/    │       │    models/      │
   │                 │       │                 │       │                 │
   │ • cleaner.py    │       │ • hybrid.py     │       │ • huggingface   │
   │ • chunker.py    │       │ • hyde.py       │       │ • openai        │
   │ • pipeline.py   │       │ • reranker.py   │       │ • anthropic     │
   └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
            │                         │                          │
            │                         ▼                          │
            │               ┌─────────────────┐                  │
            │               │   services/     │◄─────────────────┘
            │               │                 │
            │               │ • rag.py        │
            │               │ • hybrid_rag.py │
            │               └────────┬────────┘
            │                        │
            └────────────────────────┼─────────────────────────┐
                                     │                         │
                                     ▼                         ▼
                           ┌─────────────────┐       ┌─────────────────┐
                           │   evaluation/   │       │     api/        │
                           │                 │       │                 │
                           │ • harness.py    │       │ • app.py        │
                           │ • comparison.py │       │ • routes.py     │
                           │ • metrics.py    │       │                 │
                           └─────────────────┘       └─────────────────┘
```

---

## Embedding Models

| Model | Type | Use Case |
|-------|------|----------|
| `Qwen/Qwen3-Embedding-8B` | General | Default, instruction-aware |
| `abhinand/MedEmbed-large-v0.1` | Medical | Recommended for TPN domain |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` | General | Best MMTEB scores |

---

## LLM Providers

| Provider | Models | API |
|----------|--------|-----|
| HuggingFace | Qwen2.5-7B, Llama-3.1-8B, tpn-gpt-oss-20b | HF Inference API |
| OpenAI | gpt-4o, gpt-4o-mini, o1-preview | OpenAI API |
| Anthropic | claude-sonnet-4, claude-3.5-sonnet | Anthropic API |

---

## Key Configuration

```python
# Retrieval (tune empirically)
vector_weight = 0.5
bm25_weight = 0.5
rrf_k = 60

# Chunking
chunk_size = 1000
chunk_overlap = 200

# Reranker
reranker_model = "BAAI/bge-reranker-v2-m3"

# Relevance threshold
min_score_threshold = 0.0
```

---

## Quick Start Commands

```bash
# Install
pip install -e .

# Build vector store
python scripts/ingest.py --docs-dir data/documents --persist-dir ./data

# Test retrieval
python scripts/retrieve.py --demo

# Run evaluation
python eval/clinical_eval.py -n 10

# Compare models
python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct -n 50
```
