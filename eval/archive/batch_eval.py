#!/usr/bin/env python3
"""
Batch Evaluation Script for TPN RAG System.

Clean, efficient evaluation:
1. Load models ONCE at startup
2. Generate ALL answers in batch
3. Run metrics on collected results
4. Print aggregate results

Usage:
    python eval/batch_eval.py --samples 10
    python eval/batch_eval.py --samples 50
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable tokenizer parallelism warning
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@dataclass
class SampleResult:
    """Result for a single sample."""
    idx: int
    question: str
    expected: str
    phase1_answer: str
    phase2_answer: str
    context: List[str]
    sources: List[Dict]


# ============================================================================
# STEP 1: LOAD MODELS ONCE
# ============================================================================

def load_all_models():
    """Load all models once at startup."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    import torch

    print("=" * 60)
    print("LOADING MODELS (ONE TIME)")
    print("=" * 60)

    # Load LLM
    model_name = "chandramax/tpn-gpt-oss-20b"
    print(f"\n[1/4] Loading LLM: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("      LLM loaded successfully")

    # Load embedding model
    embed_name = "Qwen/Qwen3-Embedding-8B"
    print(f"\n[2/4] Loading embedding model: {embed_name}...")
    embed_model = SentenceTransformer(
        embed_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    print("      Embedding model loaded successfully")

    # Load reranker for better retrieval precision
    reranker_name = "BAAI/bge-reranker-v2-m3"
    print(f"\n[3/4] Loading reranker: {reranker_name}...")
    reranker = CrossEncoder(reranker_name, max_length=512)
    print("      Reranker loaded successfully")

    # Load ChromaDB
    print("\n[4/4] Loading ChromaDB...")
    chroma_path = project_root / "data" / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("tpn_documents")
    print(f"      ChromaDB loaded: {collection.count()} documents")

    print("\n" + "=" * 60)
    print("ALL MODELS LOADED")
    print("=" * 60)

    return {
        'llm': llm,
        'tokenizer': tokenizer,
        'embed_model': embed_model,
        'reranker': reranker,
        'collection': collection
    }


# ============================================================================
# STEP 2: GENERATE ALL ANSWERS
# ============================================================================

def load_test_samples(num_samples: int) -> List[Dict]:
    """Load test samples from JSONL file."""
    test_file = project_root / "eval" / "data" / "test_with_citations.jsonl"

    samples = []
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)

            # Extract question and expected answer
            question = None
            expected = None
            for msg in data['messages']:
                if msg['role'] == 'user':
                    question = msg['content']
                elif msg['role'] == 'assistant':
                    expected = msg.get('content', '')

            samples.append({
                'idx': i,
                'question': question,
                'expected': expected
            })

    return samples


def retrieve_context(question: str, models: Dict, initial_k: int = 20, final_k: int = 5) -> Tuple[str, List[str], List[Dict]]:
    """Retrieve context using pre-loaded models with reranking for better precision."""
    embed_model = models['embed_model']
    collection = models['collection']
    reranker = models['reranker']

    # Step 1: Initial retrieval with more candidates
    query_embedding = embed_model.encode([question], prompt_name="query")[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 2: Prepare candidates for reranking
    candidates = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        distance = results['distances'][0][i] if results['distances'] else 0
        candidates.append({
            'content': doc,
            'metadata': meta,
            'vector_score': 1 - distance
        })

    # Step 3: Rerank using cross-encoder
    pairs = [(question, c['content']) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # Combine candidates with rerank scores and sort
    for i, c in enumerate(candidates):
        c['rerank_score'] = float(rerank_scores[i])

    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Step 4: Take top_k after reranking
    context_parts = []
    context_list = []
    sources = []

    for c in candidates[:final_k]:
        source_name = c['metadata'].get('source', 'Unknown')

        context_parts.append(f"[Source: {source_name}]\n{c['content']}")
        context_list.append(c['content'])
        sources.append({
            'source': source_name,
            'vector_score': c['vector_score'],
            'rerank_score': c['rerank_score'],
            'content': c['content'][:200] + "..."
        })

    return "\n\n---\n\n".join(context_parts), context_list, sources


def generate_answer(question: str, context: str, models: Dict, max_tokens: int = 4096) -> str:
    """Generate answer using pre-loaded LLM."""
    import torch

    llm = models['llm']
    tokenizer = models['tokenizer']

    if context:
        # RAG prompt - CONCISE answers
        system_prompt = f"""You are a TPN clinical expert. Answer CONCISELY using the reference documents.

RULES:
- Be BRIEF and DIRECT (2-4 sentences max for simple questions)
- Cite sources: [Source, p.XX]
- Include specific values with units
- Only elaborate if the question requires detailed explanation

REFERENCES:
{context}

Answer concisely with citations."""
    else:
        # No RAG prompt - CONCISE
        system_prompt = """You are a TPN clinical expert. Answer CONCISELY (2-4 sentences for simple questions). Include specific values with units. Only elaborate if the question requires detailed explanation."""

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract final answer from reasoning model output
    # Look for patterns like "assistantfinal" or "Based on my analysis"
    text = response.strip()

    # If model outputs reasoning trace, extract final answer
    if 'assistantfinal' in text.lower():
        # Get everything after the last "assistantfinal"
        parts = text.lower().split('assistantfinal')
        final_idx = text.lower().rfind('assistantfinal')
        text = text[final_idx + len('assistantfinal'):].strip()
    elif 'therefore,' in text.lower():
        # Alternative: get from "Therefore" onwards
        idx = text.lower().rfind('therefore,')
        if idx > len(text) // 2:  # Only if "therefore" is in second half
            text = text[idx:]

    return text.strip()


def generate_all_answers(samples: List[Dict], models: Dict) -> List[SampleResult]:
    """Generate Phase 1 and Phase 2 answers for all samples."""
    print("\n" + "=" * 60)
    print(f"GENERATING ANSWERS FOR {len(samples)} SAMPLES")
    print("=" * 60)

    results = []

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing: {sample['question'][:60]}...")

        # Phase 1: No RAG
        print("  Phase 1 (no RAG)...")
        phase1_answer = generate_answer(sample['question'], None, models)

        # Retrieve context
        print("  Retrieving context...")
        formatted_context, context_list, sources = retrieve_context(sample['question'], models)

        # Phase 2: With RAG
        print("  Phase 2 (with RAG)...")
        phase2_answer = generate_answer(sample['question'], formatted_context, models)

        results.append(SampleResult(
            idx=sample['idx'],
            question=sample['question'],
            expected=sample['expected'],
            phase1_answer=phase1_answer,
            phase2_answer=phase2_answer,
            context=context_list,
            sources=sources
        ))

        print(f"  Done. P1: {len(phase1_answer)} chars, P2: {len(phase2_answer)} chars")

    print("\n" + "=" * 60)
    print("ALL ANSWERS GENERATED")
    print("=" * 60)

    return results


# ============================================================================
# STEP 3: RUN METRICS
# ============================================================================

def compute_metrics(results: List[SampleResult]) -> Dict:
    """Compute all metrics on collected results."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from rouge_score import rouge_scorer

    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)

    # Initialize metrics
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    clinical_metric = GEval(
        name="Clinical Correctness",
        criteria="""Evaluate clinical correctness for TPN. Check:
1. Accuracy of dosing values and units
2. Correctness of clinical protocols
3. Proper clinical reasoning""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model="gpt-5-mini",
        threshold=0.7
    )

    # Collect scores
    p1_clinical_scores = []
    p2_clinical_scores = []
    p1_rouge_scores = []
    p2_rouge_scores = []

    for i, r in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] Evaluating sample {r.idx}...")

        # ROUGE-L
        p1_rouge = rouge.score(r.expected, r.phase1_answer)['rougeL'].fmeasure
        p2_rouge = rouge.score(r.expected, r.phase2_answer)['rougeL'].fmeasure
        p1_rouge_scores.append(p1_rouge)
        p2_rouge_scores.append(p2_rouge)

        # Clinical GEval
        try:
            # Phase 1
            tc1 = LLMTestCase(input=r.question, actual_output=r.phase1_answer, expected_output=r.expected)
            clinical_metric.measure(tc1)
            p1_clinical_scores.append(clinical_metric.score)

            # Phase 2
            tc2 = LLMTestCase(input=r.question, actual_output=r.phase2_answer, expected_output=r.expected)
            clinical_metric.measure(tc2)
            p2_clinical_scores.append(clinical_metric.score)
        except Exception as e:
            print(f"  GEval error: {e}")
            p1_clinical_scores.append(0)
            p2_clinical_scores.append(0)

        print(f"  ROUGE-L: P1={p1_rouge:.3f}, P2={p2_rouge:.3f}")
        print(f"  Clinical: P1={p1_clinical_scores[-1]:.3f}, P2={p2_clinical_scores[-1]:.3f}")

    return {
        'p1_clinical': p1_clinical_scores,
        'p2_clinical': p2_clinical_scores,
        'p1_rouge': p1_rouge_scores,
        'p2_rouge': p2_rouge_scores
    }


# ============================================================================
# STEP 4: PRINT RESULTS
# ============================================================================

def print_aggregate_results(metrics: Dict, num_samples: int):
    """Print aggregate results."""

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    p1_clinical = avg(metrics['p1_clinical'])
    p2_clinical = avg(metrics['p2_clinical'])
    p1_rouge = avg(metrics['p1_rouge'])
    p2_rouge = avg(metrics['p2_rouge'])

    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({num_samples} samples)")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Phase1':>10} {'Phase2':>10} {'Delta':>10} {'RAG Lift':>10}")
    print("-" * 65)
    print(f"{'Clinical Correctness':<25} {p1_clinical:>10.3f} {p2_clinical:>10.3f} {p2_clinical - p1_clinical:>+10.3f} {((p2_clinical - p1_clinical) / p1_clinical * 100) if p1_clinical > 0 else 0:>+9.1f}%")
    print(f"{'ROUGE-L':<25} {p1_rouge:>10.3f} {p2_rouge:>10.3f} {p2_rouge - p1_rouge:>+10.3f} {((p2_rouge - p1_rouge) / p1_rouge * 100) if p1_rouge > 0 else 0:>+9.1f}%")
    print("=" * 65)

    if p2_clinical >= 0.9:
        print("\n[SUCCESS] Clinical correctness >= 90%!")
    elif p2_clinical >= 0.7:
        print("\n[GOOD] Clinical correctness >= 70%")
    else:
        print("\n[NEEDS WORK] Clinical correctness < 70%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch TPN RAG Evaluation')
    parser.add_argument('--samples', '-n', type=int, default=5, help='Number of samples')
    args = parser.parse_args()

    # Check env vars
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Step 1: Load models once
    models = load_all_models()

    # Step 2: Load samples and generate all answers
    samples = load_test_samples(args.samples)
    results = generate_all_answers(samples, models)

    # Step 3: Run metrics
    metrics = compute_metrics(results)

    # Step 4: Print results
    print_aggregate_results(metrics, args.samples)

    # Save results
    output_file = project_root / "eval" / "batch_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'num_samples': args.samples,
            'results': [asdict(r) for r in results],
            'metrics': metrics
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Cleanup
    import torch
    del models
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
