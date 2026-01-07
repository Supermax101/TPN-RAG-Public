# TPN RAG System - Progress Tracker

## 1. Project Overview

| Attribute | Value |
|-----------|-------|
| **Project** | TPN RAG System v3.0 |
| **Goal** | 90%+ accuracy on 941 grounded Q&A pairs |
| **Thesis** | Prove open models (Qwen2.5, Llama3) + RAG >= SOTA models (GPT-4, Claude) |
| **Python** | 3.11+ |
| **LangChain** | 1.2.x (Dec 2025) |

---

## 2. Completed Tasks

- [x] **Ingestion Pipeline** - 76 docs → 4,558 chunks (87% text, 13% tables)
- [x] **Document Cleaning** - OCR artifact removal, 21.4% reduction
- [x] **Semantic Chunking** - chunk_size=1000, overlap=200
- [x] **Retrieval Pipeline** - Full hybrid search implementation
  - [x] HyDE (Hypothetical Document Embeddings)
  - [x] Multi-Query expansion
  - [x] Hybrid RRF (Vector + BM25, k=60)
  - [x] Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
- [x] **Model Providers** - Unified interface for all LLMs
  - [x] HuggingFace (Qwen2.5, Llama3)
  - [x] OpenAI (GPT-4o, GPT-5-mini)
  - [x] Anthropic (Claude)
- [x] **Evaluation Framework** - Comprehensive metrics
  - [x] 941 Q&A pairs with source citations
  - [x] GEval metrics (correctness, relevance, faithfulness)
  - [x] Hit@K, F1 retrieval metrics
- [x] **Clinical-Specific Evaluation** - Deterministic value matching in `clinical_eval.py`
- [x] **Removed Ollama Dependencies** - Now HuggingFace only throughout project
- [x] **Added Embedding Model Options** - MedEmbed, KaLM, Qwen3
- [x] **Updated Judge Model** - GPT-5-mini as evaluation judge

---

## 3. In Progress

- [ ] Run evaluation on VM with different embedding models
- [ ] Compare MedEmbed vs Qwen3 vs KaLM embeddings
- [ ] Tune retrieval parameters based on results
- [ ] Optimize reranker top_k settings

---

## 4. Pending Tasks

- [ ] Establish baseline metrics (model only vs model + RAG)
- [ ] Hit 90% accuracy target
- [ ] Citation grounding improvements
- [ ] Prompt engineering iteration
- [ ] Fine-tune RRF weights (currently 0.5/0.5)
- [ ] Test reasoning model integration
- [ ] Document final architecture decisions

---

## 5. Key Files Reference

| Purpose | File Path |
|---------|-----------|
| Clinical Evaluation | `eval/clinical_eval.py` |
| Batch Evaluation | `eval/batch_eval.py` |
| Enhanced Evaluation | `eval/enhanced_eval.py` |
| Configuration | `app/config.py` |
| Retrieval Pipeline | `app/retrieval/pipeline.py` |
| Hybrid Retriever | `app/retrieval/hybrid.py` |
| HyDE Retriever | `app/retrieval/hyde.py` |
| Cross-Encoder Reranker | `app/retrieval/reranker.py` |
| Ingestion Pipeline | `app/ingestion/pipeline.py` |
| Model Factory | `app/models/__init__.py` |

---

## 6. Embedding Models to Test

| Model | Type | Notes |
|-------|------|-------|
| `Qwen/Qwen3-Embedding-8B` | General | Current default |
| `abhinand/MedEmbed-large-v0.1` | Medical | Recommended for TPN domain |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` | General | Best MMTEB scores |

### Selection in CLI

```bash
# 1 = Qwen3-Embedding-8B (default)
# 2 = MedEmbed-large-v0.1 (medical)
# 3 = KaLM-Embedding-Gemma3-12B
```

---

## 7. Commands Quick Reference

```bash
# === Installation ===
pip install -e .

# === Document Ingestion ===
python scripts/ingest.py --docs-dir data/documents --persist-dir ./data
python scripts/ingest.py --no-vector-store  # BM25 only

# === Retrieval Testing ===
python scripts/retrieve.py --demo
python scripts/retrieve.py --persist-dir ./data --query "protein requirements"

# === Clinical Evaluation ===
python eval/clinical_eval.py -n 10 --embed-model 2

# === Batch Evaluation ===
python eval/batch_eval.py --samples 50

# === Model Comparison ===
python scripts/compare_models.py --list-models
python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct -n 50
python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct openai:gpt-4o-mini -n 100

# === Standard Evaluation ===
python scripts/evaluate.py --analyze-only
python scripts/evaluate.py -n 100

# === Development ===
black app eval cli.py scripts
ruff check app eval cli.py scripts
pytest
```

---

## 8. Configuration Reference

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

## 9. Data Paths

| Data | Location |
|------|----------|
| Source documents | `data/documents/` (76 MD files) |
| Grounded Q&A | `/Users/chandra/Desktop/TPN2.OFinetuning/data/final/test.jsonl` |
| Persisted indexes | `./data/chroma/`, `./data/bm25/` |
| Comparison results | `./comparison_results/` |

---

## 10. Session Log

### 2026-01-07
- Created PROGRESS_TRACKER.md
- Completed full Ollama cleanup across codebase
- Updated all evaluation scripts to use GPT-5-mini as judge
- Added MedEmbed, KaLM embedding model options
- Current state: All pipelines implemented, ready for VM evaluation
- Next steps: Run embedding model comparison on VM

---

## 11. Metrics Tracking

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Overall Accuracy | TBD | TBD | 90%+ |
| Hit@5 | TBD | TBD | 95%+ |
| Hit@10 | TBD | TBD | 98%+ |
| Answer F1 | TBD | TBD | 0.85+ |

---

## 12. Architecture Diagram

```
INGESTION:   DPT2 Docs → Cleaner → Chunker → ChromaDB + BM25
RETRIEVAL:   Query → [HyDE] → [Multi-Query] → Hybrid Search → Reranker
GENERATION:  Context + Prompt → LLM (HuggingFace/OpenAI/Anthropic) → Answer
EVALUATION:  941 Q&A pairs → Metrics (Hit@K, F1) → RAG Lift Analysis
```

---

## Notes

- All Ollama dependencies have been removed; use HuggingFace for local models
- MedEmbed recommended for medical domain (TPN-specific terminology)
- Cross-encoder reranker significantly improves precision
- top_k increased to 10 for reasoning model compatibility
