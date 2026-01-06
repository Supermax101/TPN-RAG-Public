"""
Enhanced Two-Phase Evaluation Script with Comprehensive Metrics.

Improvements over sample_eval.py:
1. max_new_tokens increased from 512 to 1024
2. Added BLEU-1, BLEU-4, ROUGE-L deterministic metrics
3. Added Faithfulness metric (is answer grounded in RAG context?)
4. Added Contextual Recall metric (did retrieval find all needed info?)
5. Clinical-specific GEval with explicit rubric for dosing/units/ranges
6. Component-level evaluation (retrieval quality separate from generation)

Usage:
    python eval/enhanced_eval.py                    # Run on 1 sample
    python eval/enhanced_eval.py --samples 10       # Run on 10 samples
    python eval/enhanced_eval.py --samples 50       # Run on 50 samples
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import re

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables before imports
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@dataclass
class MetricResult:
    """Container for all metric results."""
    # Deterministic metrics
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    rouge_l: float = 0.0
    f1_score: float = 0.0
    key_phrase_overlap: float = 0.0

    # LLM-based metrics
    clinical_correctness: float = 0.0
    clinical_reason: str = ""
    faithfulness: float = 0.0
    faithfulness_reason: str = ""
    answer_relevancy: float = 0.0

    # Retrieval metrics
    contextual_recall: float = 0.0
    contextual_precision: float = 0.0
    avg_retrieval_score: float = 0.0


@dataclass
class EvaluationResult:
    """Full evaluation result for one sample."""
    idx: int
    question: str
    expected_answer: str
    phase1_answer: str
    phase2_answer: str
    retrieval_context: List[str]
    sources: List[Dict]
    phase1_metrics: MetricResult
    phase2_metrics: MetricResult
    retrieval_metrics: Dict


# ============================================================================
# DETERMINISTIC METRICS (BLEU, ROUGE, F1)
# ============================================================================

def compute_bleu_scores(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Compute BLEU-1 and BLEU-4 scores."""
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk

        # Ensure nltk data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        # Tokenize
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0, 0.0

        # Use smoothing to handle zero counts
        smoother = SmoothingFunction().method1

        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

        return bleu_1, bleu_4
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0, 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"ROUGE-L calculation error: {e}")
        return 0.0


def compute_f1_score(reference: str, hypothesis: str) -> float:
    """Compute token-level F1 score."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = ref_tokens & hyp_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def extract_clinical_values(text: str) -> List[str]:
    """Extract clinical values (dosing, ranges, units) from text."""
    patterns = [
        r'\d+\.?\d*\s*(?:mg|g|mcg|mL|L|mEq|mmol|IU)/(?:kg|day|hr|min|L)',  # Dosing
        r'\d+\.?\d*\s*(?:to|-)\s*\d+\.?\d*\s*(?:mg|g|mcg|mL|%)',  # Ranges
        r'\d+\.?\d*\s*(?:mg|g|mcg|mL|L|mEq|mmol|IU|kcal)',  # Values with units
        r'\d+\.?\d*\s*%',  # Percentages
    ]

    values = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        values.extend(matches)

    return [v.lower().strip() for v in values]


def compute_key_phrase_overlap(reference: str, hypothesis: str) -> float:
    """Compute overlap of clinical key phrases."""
    ref_values = set(extract_clinical_values(reference))
    hyp_values = set(extract_clinical_values(hypothesis))

    if len(ref_values) == 0:
        return 1.0 if len(hyp_values) == 0 else 0.5

    overlap = len(ref_values & hyp_values)
    return overlap / len(ref_values)


def compute_deterministic_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute all deterministic metrics."""
    bleu_1, bleu_4 = compute_bleu_scores(reference, hypothesis)
    rouge_l = compute_rouge_l(reference, hypothesis)
    f1 = compute_f1_score(reference, hypothesis)
    key_phrase = compute_key_phrase_overlap(reference, hypothesis)

    return {
        'bleu_1': bleu_1,
        'bleu_4': bleu_4,
        'rouge_l': rouge_l,
        'f1_score': f1,
        'key_phrase_overlap': key_phrase
    }


# ============================================================================
# LLM-BASED METRICS (DeepEval)
# ============================================================================

def compute_clinical_geval(question: str, expected: str, actual: str) -> Tuple[float, str]:
    """Compute clinical correctness using GEval with clinical-specific rubric."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        clinical_metric = GEval(
            name="Clinical Correctness",
            criteria="""Evaluate clinical correctness for TPN (Total Parenteral Nutrition) with emphasis on:

1. DOSING VALUES: Are numeric values exactly correct? (e.g., "3-4 g/kg/day" must match)
2. UNITS: Are units correct? (mg vs g, mEq vs mmol, /kg vs /day)
3. RANGES: Are ranges accurate? (e.g., "2-3 mEq/kg" not "1-5 mEq/kg")
4. POPULATIONS: Is the patient population correct? (preterm vs term, neonatal vs pediatric)
5. PROTOCOLS: Are clinical procedures/protocols accurate and complete?
6. SAFETY: Are contraindications and monitoring requirements mentioned?""",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            evaluation_steps=[
                "Extract all numeric values, units, and ranges from both expected and actual answers",
                "Compare each clinical value for exact match or clinically acceptable equivalence",
                "Check if the patient population (age group, condition) is correctly specified",
                "Verify that protocol steps are complete and in the correct order",
                "Assess if safety information (contraindications, monitoring) is included",
                "Assign score: 9-10=exact match, 7-8=minor issues, 5-6=some errors, 3-4=major errors, 0-2=dangerous/wrong"
            ],
            model="gpt-4o-mini",
            threshold=0.7
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual,
            expected_output=expected
        )

        clinical_metric.measure(test_case)
        return clinical_metric.score, clinical_metric.reason or ""

    except Exception as e:
        print(f"Clinical GEval error: {e}")
        return 0.0, str(e)


def compute_faithfulness(question: str, actual: str, retrieval_context: List[str]) -> Tuple[float, str]:
    """Compute faithfulness - is the answer grounded in retrieved context?"""
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        faithfulness_metric = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4o-mini",
            include_reason=True
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual,
            retrieval_context=retrieval_context
        )

        faithfulness_metric.measure(test_case)
        return faithfulness_metric.score, faithfulness_metric.reason or ""

    except Exception as e:
        print(f"Faithfulness error: {e}")
        return 0.0, str(e)


def compute_answer_relevancy(question: str, actual: str) -> float:
    """Compute answer relevancy - does the answer address the question?"""
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4o-mini"
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual
        )

        relevancy_metric.measure(test_case)
        return relevancy_metric.score

    except Exception as e:
        print(f"Answer relevancy error: {e}")
        return 0.0


def compute_contextual_recall(question: str, expected: str, retrieval_context: List[str]) -> float:
    """Compute contextual recall - did we retrieve all info needed for expected answer?"""
    try:
        from deepeval.metrics import ContextualRecallMetric
        from deepeval.test_case import LLMTestCase

        recall_metric = ContextualRecallMetric(
            threshold=0.7,
            model="gpt-4o-mini"
        )

        test_case = LLMTestCase(
            input=question,
            expected_output=expected,
            retrieval_context=retrieval_context
        )

        recall_metric.measure(test_case)
        return recall_metric.score

    except Exception as e:
        print(f"Contextual recall error: {e}")
        return 0.0


def compute_contextual_precision(question: str, expected: str, retrieval_context: List[str]) -> float:
    """Compute contextual precision - are retrieved docs relevant to the question?"""
    try:
        from deepeval.metrics import ContextualPrecisionMetric
        from deepeval.test_case import LLMTestCase

        precision_metric = ContextualPrecisionMetric(
            threshold=0.7,
            model="gpt-4o-mini"
        )

        test_case = LLMTestCase(
            input=question,
            expected_output=expected,
            retrieval_context=retrieval_context
        )

        precision_metric.measure(test_case)
        return precision_metric.score

    except Exception as e:
        print(f"Contextual precision error: {e}")
        return 0.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_single_sample(test_file: str, sample_idx: int = 0) -> dict:
    """Load a single sample from the JSONL test file."""
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise IndexError(f"Sample index {sample_idx} not found in {test_file}")


def extract_qa_from_sample(sample: dict) -> Tuple[str, str, str]:
    """Extract question and expected answer from sample."""
    messages = sample['messages']

    question = None
    expected_answer = None
    expected_thinking = None

    for msg in messages:
        if msg['role'] == 'user':
            question = msg['content']
        elif msg['role'] == 'assistant':
            expected_answer = msg.get('content', '')
            expected_thinking = msg.get('thinking', '')

    return question, expected_answer, expected_thinking


# ============================================================================
# MODEL MANAGER - Load models once and reuse
# ============================================================================

class ModelManager:
    """Singleton to manage model loading - load once, reuse everywhere."""

    _instance = None
    _llm_model = None
    _llm_tokenizer = None
    _embed_model = None
    _chroma_collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_llm(self, model_name: str = "chandramax/tpn-gpt-oss-20b"):
        """Load LLM model once."""
        if self._llm_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"[ModelManager] Loading LLM: {model_name}...")
            self._llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("[ModelManager] LLM loaded successfully")
        return self._llm_model, self._llm_tokenizer

    def load_embedding_model(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        """Load embedding model once."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            print(f"[ModelManager] Loading embedding model: {model_name}...")
            self._embed_model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            print("[ModelManager] Embedding model loaded successfully")
        return self._embed_model

    def load_chroma_collection(self):
        """Load ChromaDB collection once."""
        if self._chroma_collection is None:
            import chromadb

            chroma_path = project_root / "data" / "chroma"
            client = chromadb.PersistentClient(path=str(chroma_path))
            self._chroma_collection = client.get_collection("tpn_documents")
            print(f"[ModelManager] ChromaDB loaded: {self._chroma_collection.count()} documents")
        return self._chroma_collection

    def cleanup(self):
        """Free GPU memory."""
        import torch

        if self._llm_model is not None:
            del self._llm_model
            self._llm_model = None
        if self._llm_tokenizer is not None:
            del self._llm_tokenizer
            self._llm_tokenizer = None
        if self._embed_model is not None:
            del self._embed_model
            self._embed_model = None

        torch.cuda.empty_cache()
        print("[ModelManager] Cleaned up GPU memory")


# Global model manager instance
model_manager = ModelManager()


# ============================================================================
# RETRIEVAL
# ============================================================================

def retrieve_context(question: str, top_k: int = 5) -> Tuple[str, List[str], List[Dict]]:
    """
    Retrieve relevant context from ChromaDB.
    Returns: (formatted_context, list_of_context_strings, sources_metadata)
    """
    embed_model = model_manager.load_embedding_model()
    collection = model_manager.load_chroma_collection()

    query_embedding = embed_model.encode([question], prompt_name="query")[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []
    context_list = []
    sources = []

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        distance = results['distances'][0][i] if results['distances'] else 0

        source_name = meta.get('source', meta.get('document_name', 'Unknown'))
        page_num = meta.get('page_num', '')
        section = meta.get('section', 'General')
        score = 1 - distance

        source_ref = f"{source_name}"
        if page_num:
            source_ref += f", Page {page_num}"

        context_parts.append(f"[Source: {source_ref}]\nSection: {section}\n{doc}")
        context_list.append(doc)
        sources.append({
            'source': source_name,
            'page': page_num,
            'section': section,
            'score': score,
            'content': doc[:200] + "..."
        })

    return "\n\n---\n\n".join(context_parts), context_list, sources


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def run_model_inference(
    question: str,
    context: Optional[str] = None,
    max_new_tokens: int = 1024
) -> str:
    """Run model inference with or without RAG context. Uses pre-loaded model."""
    import torch

    model, tokenizer = model_manager.load_llm()

    if context:
        # Hospital-Grade RAG Prompt
        system_prompt = f"""<role>
You are a clinical expert specializing in neonatal and pediatric Total Parenteral Nutrition (TPN).
You have been fine-tuned on evidence-based TPN guidelines and are now augmented with real-time reference documents.
</role>

<source_priority>
When answering, prioritize information in this order:
1. REFERENCE DOCUMENTS (below) - Most authoritative for specific values
2. Your trained clinical knowledge - For general principles and reasoning
3. If sources conflict, cite BOTH and note the discrepancy
</source_priority>

<instructions>
1. ALWAYS cite specific values from references: [Source Name, p.XX: "exact value"]
2. Use your trained knowledge for clinical reasoning and context
3. Show step-by-step reasoning using this structure:
   - CLINICAL CONTEXT: Patient factors and considerations
   - EVIDENCE: What the references state (with citations)
   - REASONING: How to apply this to the clinical scenario
   - RECOMMENDATION: Final answer with specific values and units
4. Include units for ALL doses (g/kg/day, mEq/kg/day, mg/kg/min, etc.)
5. Flag any safety concerns with ⚠️ WARNING
</instructions>

<constraints>
- NEVER fabricate values not in references or training
- If references are insufficient, state: "Reference documents do not contain this information"
- If values seem clinically unsafe, flag for verification
- Note when specialist consultation is recommended
- Do not extrapolate pediatric doses from adult data without explicit guidance
</constraints>

<conflict_resolution>
If reference documents contain conflicting values:
- Present BOTH values with their sources
- Note institutional variation if applicable
- Recommend using your institution's protocol as the tiebreaker
</conflict_resolution>

<reference_documents>
{context}
</reference_documents>

Provide your clinical guidance using both the reference documents and your trained expertise. Cite all specific values."""
    else:
        # Base prompt for model-only inference (no RAG)
        system_prompt = """You are a clinical expert specializing in neonatal and pediatric Total Parenteral Nutrition (TPN). Provide accurate, evidence-based guidance for TPN management including dosing calculations, monitoring protocols, and complication management. Always show your reasoning step-by-step."""

    # Use "developer" role to match fine-tuning format
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # INCREASED
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response.strip()


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def evaluate_single_sample(
    idx: int,
    question: str,
    expected: str,
    verbose: bool = True
) -> EvaluationResult:
    """Run full evaluation on a single sample."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"SAMPLE {idx + 1}")
        print(f"{'='*70}")
        print(f"Q: {question[:100]}...")

    # Phase 1: Model only (no RAG)
    if verbose:
        print("\n[Phase 1] Running model without RAG...")
    phase1_answer = run_model_inference(question, context=None)

    # Retrieve context
    if verbose:
        print("[Retrieval] Getting relevant documents...")
    formatted_context, context_list, sources = retrieve_context(question, top_k=5)

    avg_retrieval_score = sum(s['score'] for s in sources) / len(sources) if sources else 0
    if verbose:
        print(f"  Retrieved {len(sources)} docs, avg score: {avg_retrieval_score:.3f}")

    # Phase 2: Model with RAG
    if verbose:
        print("[Phase 2] Running model with RAG context...")
    phase2_answer = run_model_inference(question, context=formatted_context)

    # =========== COMPUTE METRICS ===========

    if verbose:
        print("\n[Metrics] Computing deterministic metrics...")

    # Deterministic metrics for Phase 1
    p1_det = compute_deterministic_metrics(expected, phase1_answer)

    # Deterministic metrics for Phase 2
    p2_det = compute_deterministic_metrics(expected, phase2_answer)

    if verbose:
        print("[Metrics] Computing LLM-based metrics...")

    # Clinical GEval for both phases
    p1_clinical, p1_clinical_reason = compute_clinical_geval(question, expected, phase1_answer)
    p2_clinical, p2_clinical_reason = compute_clinical_geval(question, expected, phase2_answer)

    # Faithfulness (only for Phase 2 - needs retrieval context)
    p2_faithfulness, p2_faith_reason = compute_faithfulness(question, phase2_answer, context_list)

    # Answer relevancy
    p1_relevancy = compute_answer_relevancy(question, phase1_answer)
    p2_relevancy = compute_answer_relevancy(question, phase2_answer)

    # Retrieval quality metrics
    if verbose:
        print("[Metrics] Computing retrieval quality...")
    contextual_recall = compute_contextual_recall(question, expected, context_list)
    contextual_precision = compute_contextual_precision(question, expected, context_list)

    # Build result objects
    phase1_metrics = MetricResult(
        bleu_1=p1_det['bleu_1'],
        bleu_4=p1_det['bleu_4'],
        rouge_l=p1_det['rouge_l'],
        f1_score=p1_det['f1_score'],
        key_phrase_overlap=p1_det['key_phrase_overlap'],
        clinical_correctness=p1_clinical,
        clinical_reason=p1_clinical_reason[:200] if p1_clinical_reason else "",
        faithfulness=0.0,  # N/A for phase 1
        faithfulness_reason="",
        answer_relevancy=p1_relevancy
    )

    phase2_metrics = MetricResult(
        bleu_1=p2_det['bleu_1'],
        bleu_4=p2_det['bleu_4'],
        rouge_l=p2_det['rouge_l'],
        f1_score=p2_det['f1_score'],
        key_phrase_overlap=p2_det['key_phrase_overlap'],
        clinical_correctness=p2_clinical,
        clinical_reason=p2_clinical_reason[:200] if p2_clinical_reason else "",
        faithfulness=p2_faithfulness,
        faithfulness_reason=p2_faith_reason[:200] if p2_faith_reason else "",
        answer_relevancy=p2_relevancy
    )

    retrieval_metrics = {
        'contextual_recall': contextual_recall,
        'contextual_precision': contextual_precision,
        'avg_retrieval_score': avg_retrieval_score
    }

    if verbose:
        print("\n" + "-"*50)
        print("RESULTS:")
        print("-"*50)
        print(f"{'Metric':<25} {'Phase1':>10} {'Phase2':>10} {'Delta':>10}")
        print("-"*50)
        print(f"{'Clinical Correctness':<25} {p1_clinical:>10.3f} {p2_clinical:>10.3f} {p2_clinical - p1_clinical:>+10.3f}")
        print(f"{'BLEU-1':<25} {p1_det['bleu_1']:>10.3f} {p2_det['bleu_1']:>10.3f} {p2_det['bleu_1'] - p1_det['bleu_1']:>+10.3f}")
        print(f"{'BLEU-4':<25} {p1_det['bleu_4']:>10.3f} {p2_det['bleu_4']:>10.3f} {p2_det['bleu_4'] - p1_det['bleu_4']:>+10.3f}")
        print(f"{'ROUGE-L':<25} {p1_det['rouge_l']:>10.3f} {p2_det['rouge_l']:>10.3f} {p2_det['rouge_l'] - p1_det['rouge_l']:>+10.3f}")
        print(f"{'F1 Score':<25} {p1_det['f1_score']:>10.3f} {p2_det['f1_score']:>10.3f} {p2_det['f1_score'] - p1_det['f1_score']:>+10.3f}")
        print(f"{'Key Phrase Overlap':<25} {p1_det['key_phrase_overlap']:>10.3f} {p2_det['key_phrase_overlap']:>10.3f} {p2_det['key_phrase_overlap'] - p1_det['key_phrase_overlap']:>+10.3f}")
        print(f"{'Answer Relevancy':<25} {p1_relevancy:>10.3f} {p2_relevancy:>10.3f} {p2_relevancy - p1_relevancy:>+10.3f}")
        print(f"{'Faithfulness (P2 only)':<25} {'N/A':>10} {p2_faithfulness:>10.3f}")
        print("-"*50)
        print(f"{'Contextual Recall':<25} {contextual_recall:>10.3f}")
        print(f"{'Contextual Precision':<25} {contextual_precision:>10.3f}")
        print(f"{'Avg Retrieval Score':<25} {avg_retrieval_score:>10.3f}")

    return EvaluationResult(
        idx=idx,
        question=question,
        expected_answer=expected,
        phase1_answer=phase1_answer,
        phase2_answer=phase2_answer,
        retrieval_context=context_list,
        sources=sources,
        phase1_metrics=phase1_metrics,
        phase2_metrics=phase2_metrics,
        retrieval_metrics=retrieval_metrics
    )


def run_evaluation(num_samples: int = 1, verbose: bool = True) -> List[EvaluationResult]:
    """Run evaluation on multiple samples."""

    test_file = project_root / "eval" / "data" / "test_with_citations.jsonl"
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)

    print("="*70)
    print(f"ENHANCED EVALUATION - {num_samples} SAMPLES")
    print("="*70)
    print("Metrics included:")
    print("  - Deterministic: BLEU-1, BLEU-4, ROUGE-L, F1, Key Phrase Overlap")
    print("  - LLM-based: Clinical GEval, Faithfulness, Answer Relevancy")
    print("  - Retrieval: Contextual Recall, Contextual Precision")
    print("  - max_new_tokens: 1024 (increased from 512)")
    print("="*70)

    results = []

    for idx in range(num_samples):
        try:
            sample = load_single_sample(str(test_file), sample_idx=idx)
            question, expected, _ = extract_qa_from_sample(sample)

            result = evaluate_single_sample(idx, question, expected, verbose=verbose)
            results.append(result)

        except Exception as e:
            print(f"\nERROR on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def print_aggregate_results(results: List[EvaluationResult]):
    """Print aggregate statistics across all samples."""

    if not results:
        print("No results to aggregate.")
        return

    n = len(results)

    # Aggregate Phase 1 metrics
    p1_clinical = sum(r.phase1_metrics.clinical_correctness for r in results) / n
    p1_bleu1 = sum(r.phase1_metrics.bleu_1 for r in results) / n
    p1_bleu4 = sum(r.phase1_metrics.bleu_4 for r in results) / n
    p1_rouge = sum(r.phase1_metrics.rouge_l for r in results) / n
    p1_f1 = sum(r.phase1_metrics.f1_score for r in results) / n
    p1_kp = sum(r.phase1_metrics.key_phrase_overlap for r in results) / n
    p1_rel = sum(r.phase1_metrics.answer_relevancy for r in results) / n

    # Aggregate Phase 2 metrics
    p2_clinical = sum(r.phase2_metrics.clinical_correctness for r in results) / n
    p2_bleu1 = sum(r.phase2_metrics.bleu_1 for r in results) / n
    p2_bleu4 = sum(r.phase2_metrics.bleu_4 for r in results) / n
    p2_rouge = sum(r.phase2_metrics.rouge_l for r in results) / n
    p2_f1 = sum(r.phase2_metrics.f1_score for r in results) / n
    p2_kp = sum(r.phase2_metrics.key_phrase_overlap for r in results) / n
    p2_rel = sum(r.phase2_metrics.answer_relevancy for r in results) / n
    p2_faith = sum(r.phase2_metrics.faithfulness for r in results) / n

    # Retrieval metrics
    ctx_recall = sum(r.retrieval_metrics['contextual_recall'] for r in results) / n
    ctx_precision = sum(r.retrieval_metrics['contextual_precision'] for r in results) / n
    avg_ret = sum(r.retrieval_metrics['avg_retrieval_score'] for r in results) / n

    print("\n" + "="*70)
    print(f"AGGREGATE RESULTS ({n} samples)")
    print("="*70)
    print(f"\n{'Metric':<25} {'Phase1':>10} {'Phase2':>10} {'Delta':>10} {'RAG Lift':>10}")
    print("-"*65)
    print(f"{'Clinical Correctness':<25} {p1_clinical:>10.3f} {p2_clinical:>10.3f} {p2_clinical - p1_clinical:>+10.3f} {((p2_clinical - p1_clinical) / p1_clinical * 100) if p1_clinical > 0 else 0:>+9.1f}%")
    print(f"{'BLEU-1':<25} {p1_bleu1:>10.3f} {p2_bleu1:>10.3f} {p2_bleu1 - p1_bleu1:>+10.3f} {((p2_bleu1 - p1_bleu1) / p1_bleu1 * 100) if p1_bleu1 > 0 else 0:>+9.1f}%")
    print(f"{'BLEU-4':<25} {p1_bleu4:>10.3f} {p2_bleu4:>10.3f} {p2_bleu4 - p1_bleu4:>+10.3f} {((p2_bleu4 - p1_bleu4) / p1_bleu4 * 100) if p1_bleu4 > 0 else 0:>+9.1f}%")
    print(f"{'ROUGE-L':<25} {p1_rouge:>10.3f} {p2_rouge:>10.3f} {p2_rouge - p1_rouge:>+10.3f} {((p2_rouge - p1_rouge) / p1_rouge * 100) if p1_rouge > 0 else 0:>+9.1f}%")
    print(f"{'F1 Score':<25} {p1_f1:>10.3f} {p2_f1:>10.3f} {p2_f1 - p1_f1:>+10.3f} {((p2_f1 - p1_f1) / p1_f1 * 100) if p1_f1 > 0 else 0:>+9.1f}%")
    print(f"{'Key Phrase Overlap':<25} {p1_kp:>10.3f} {p2_kp:>10.3f} {p2_kp - p1_kp:>+10.3f} {((p2_kp - p1_kp) / p1_kp * 100) if p1_kp > 0 else 0:>+9.1f}%")
    print(f"{'Answer Relevancy':<25} {p1_rel:>10.3f} {p2_rel:>10.3f} {p2_rel - p1_rel:>+10.3f} {((p2_rel - p1_rel) / p1_rel * 100) if p1_rel > 0 else 0:>+9.1f}%")
    print(f"{'Faithfulness':<25} {'N/A':>10} {p2_faith:>10.3f}")

    print("\n" + "-"*65)
    print("RETRIEVAL QUALITY:")
    print("-"*65)
    print(f"{'Contextual Recall':<25} {ctx_recall:>10.3f}")
    print(f"{'Contextual Precision':<25} {ctx_precision:>10.3f}")
    print(f"{'Avg Retrieval Score':<25} {avg_ret:>10.3f}")

    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)

    # Diagnose issues
    if ctx_recall < 0.7:
        print("  [!] LOW CONTEXTUAL RECALL - Retrieval is missing needed information")
        print("      Action: Improve chunking, add more documents, or tune embeddings")

    if ctx_precision < 0.7:
        print("  [!] LOW CONTEXTUAL PRECISION - Retrieved docs not relevant enough")
        print("      Action: Improve query expansion or reranking")

    if p2_faith < 0.7:
        print("  [!] LOW FAITHFULNESS - Model ignoring RAG context")
        print("      Action: Improve prompt to emphasize using provided context")

    if p2_clinical < 0.7:
        print("  [!] LOW CLINICAL CORRECTNESS - Answers still inaccurate")
        print("      Action: Review model outputs for specific error patterns")

    if (p2_clinical - p1_clinical) < 0.1:
        print("  [!] LOW RAG LIFT - RAG not helping much")
        print("      Action: Check if retrieval is finding the right content")

    if p2_clinical >= 0.9:
        print("  [✓] TARGET ACHIEVED - Clinical correctness >= 90%!")


def save_results(results: List[EvaluationResult], output_path: Path):
    """Save results to JSON file."""

    output_data = {
        'num_samples': len(results),
        'results': [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced TPN RAG Evaluation')
    parser.add_argument('--samples', '-n', type=int, default=1, help='Number of samples to evaluate')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()

    # Check for required env vars
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Required for LLM-based metrics.")
        print("Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    # Run evaluation
    results = run_evaluation(num_samples=args.samples, verbose=not args.quiet)

    # Print aggregate results
    print_aggregate_results(results)

    # Save results
    output_path = project_root / "eval" / "enhanced_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
