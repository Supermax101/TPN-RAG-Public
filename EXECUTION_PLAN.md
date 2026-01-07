# TPN RAG Evaluation - Execution Plan

## Project Overview

**Goal:** Evaluate a fine-tuned TPN (Total Parenteral Nutrition) model with and without RAG to prove that RAG improves clinical accuracy and citation quality.

**Fine-tuned Model:** `chandramax/tpn-gpt-oss-20b` (HuggingFace)

**Test Dataset:** 941 grounded Q&A pairs in `eval/data/test_with_citations.jsonl`

---

## Two-Phase Evaluation Strategy

| Phase | Description | Purpose |
|-------|-------------|---------|
| **Phase 1** | Fine-tuned model ONLY (no RAG context) | Baseline - what the model knows on its own |
| **Phase 2** | Fine-tuned model + RAG (ChromaDB retrieval) | Full system - should show improvement |

**Metrics (using DeepEval GEval with GPT-4o-mini as judge):**
- Clinical Correctness (dosing, values, ranges, protocols)
- Citation Quality (proper source attribution)
- RAG Lift = (Phase2 - Phase1) / Phase1

---

## Current Status

### Completed:
1. ✅ Fixed module import errors (`models.py` → `data_models.py`)
2. ✅ Created `eval/sample_eval.py` - two-phase evaluation script
3. ✅ Added test datasets to `eval/data/`
4. ✅ Pushed all changes to GitHub (`Supermax101/TPN-RAG-Public`)

### Next Steps (ON VM):
1. ⏳ Pull latest changes on VM
2. ⏳ Run `sample_eval.py` on 1 sample first
3. ⏳ Analyze results, iterate if needed
4. ⏳ Scale to full 941 samples

---

## VM Setup Requirements

### Environment Variables (set these on VM):
```bash
export HF_TOKEN=your_huggingface_token_here
export OPENAI_API_KEY=your_openai_api_key_here
```

### VM Should Have:
- H200 GPU (vast.ai instance)
- ChromaDB with 7,697 indexed TPN document chunks
- Python environment with: transformers, sentence-transformers, chromadb, deepeval
- Qwen3-Embedding-8B for retrieval

---

## Execution Commands (Run on VM)

### Step 1: Pull Latest Code
```bash
cd /root/TPN-RAG-Public
git pull origin main
```

### Step 2: Verify Environment
```bash
# Check env vars are set
echo $HF_TOKEN
echo $OPENAI_API_KEY

# Check ChromaDB exists
ls -la data/chroma/
```

### Step 3: Run Sample Evaluation (1 sample)
```bash
python eval/sample_eval.py
```

This will:
1. Load sample #0 from test_with_citations.jsonl
2. Run Phase 1: Model without RAG
3. Run Phase 2: Model with RAG (retrieves from ChromaDB)
4. Run GEval comparison
5. Save results to `eval/sample_result.json`

### Step 4: Analyze Results
Check the output for:
- Phase 1 Clinical Correctness score
- Phase 2 Clinical Correctness score
- Delta (should be positive = RAG helps)
- Citation Quality improvement

### Step 5: If Results Look Good, Scale Up
Create a full evaluation script or modify sample_eval.py to loop through all 941 samples.

---

## Key Files

| File | Purpose |
|------|---------|
| `eval/sample_eval.py` | Two-phase evaluation script |
| `eval/data/test_with_citations.jsonl` | 941 Q&A pairs with citations |
| `eval/data/test_without_citations.jsonl` | Same Q&A without citations |
| `app/data_models.py` | Pydantic models (renamed from models.py) |
| `app/services/rag.py` | Core RAG service |
| `app/providers/vectorstore.py` | ChromaDB integration |

---

## Test Data Format

Each sample in `test_with_citations.jsonl`:
```json
{
  "messages": [
    {"role": "developer", "content": "System prompt..."},
    {"role": "user", "content": "Clinical question..."},
    {"role": "assistant", "thinking": "Step-by-step reasoning...", "content": "Answer with [citations]"}
  ]
}
```

---

## Expected Output

```
PHASE 1 (Model Only):
  Clinical Correctness: 0.XXX
  Citation Quality: 0.XXX

PHASE 2 (Model + RAG):
  Clinical Correctness: 0.XXX
  Citation Quality: 0.XXX

COMPARISON:
  Clinical Delta: +X.XXX (RAG improves accuracy)
  Citation Delta: +X.XXX (RAG grounds citations)
```

---

## Troubleshooting

### If ChromaDB not found:
```bash
# Re-index documents (takes time)
python -c "from app.services.loader import DocumentLoader; ..."
```

### If HF model fails to load:
```bash
# Check GPU memory
nvidia-smi

# Try with lower precision
# Edit sample_eval.py: torch_dtype=torch.float16
```

### If GEval fails:
```bash
# Verify OpenAI key
python -c "import openai; print(openai.api_key)"
```

---

## Architecture Reference

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│ Phase 1: Model Only                 │
│ - Load chandramax/tpn-gpt-oss-20b   │
│ - Generate answer (no context)      │
│ - Save response                     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Phase 2: Model + RAG                │
│ - Retrieve from ChromaDB            │
│ - Build context from chunks         │
│ - Generate answer with context      │
│ - Ground citations to real sources  │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ GEval Comparison                    │
│ - GPT-4o-mini as judge              │
│ - Score Clinical Correctness        │
│ - Score Citation Quality            │
│ - Calculate RAG Lift                │
└─────────────────────────────────────┘
```

---

## Contact/Context

- **Private Repo:** `Takeoff41-Inc/TPN2.0RAG` (also updated)
- **Public Repo:** `Supermax101/TPN-RAG-Public` (use this on VM)
- **VM Platform:** vast.ai with H200 GPU
- **Embedding Model:** `Qwen/Qwen3-Embedding-8B`

---

## Summary for Next Agent

1. You are in `/Users/chandra/Documents/TPN-RAG-Public`
2. Code is pushed to GitHub, needs to be pulled on VM
3. Run `eval/sample_eval.py` on VM to test 1 sample
4. Analyze results, then scale to full 941 samples
5. Goal: Prove RAG improves clinical accuracy over baseline
