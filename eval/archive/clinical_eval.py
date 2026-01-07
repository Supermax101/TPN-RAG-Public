#!/usr/bin/env python3
"""
Clinical-Specific TPN RAG Evaluation.

This evaluation script is designed specifically for clinical correctness:
1. Extract and compare EXACT clinical values (doses, units, ranges)
2. Separate retrieval quality from generation quality
3. Use deterministic matching for clinical values, not LLM-as-judge
4. Provide actionable diagnostics

Key Insight: For clinical evaluation, we need to:
- Check if EXACT values match (3-4 g/kg/day == 3-4 g/kg/day)
- Check if UNITS are correct (mg vs g, mEq vs mmol)
- Check if the RIGHT SOURCE was retrieved (Hit@K)
- Separately evaluate: retrieval worked? model used context?

Usage:
    python eval/clinical_eval.py --samples 10
    python eval/clinical_eval.py --samples 50 --diagnose
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import Counter

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ============================================================================
# CLINICAL VALUE EXTRACTION
# ============================================================================

@dataclass
class ClinicalValue:
    """A clinical value extracted from text."""
    raw: str                    # Original matched text
    value: str                  # Numeric value(s)
    unit: str                   # Unit (g, mg, mEq, etc.)
    denominator: str = ""       # Per what (kg, day, hr, etc.)
    is_range: bool = False      # Is it a range (e.g., 3-4)?

    def normalized(self) -> str:
        """Normalize for comparison."""
        # Convert to lowercase and standardize
        norm = self.raw.lower()
        norm = norm.replace(" ", "")
        norm = norm.replace("to", "-")
        norm = norm.replace("–", "-")  # en-dash
        return norm


class ClinicalValueExtractor:
    """Extract clinical values from text for comparison."""

    # Patterns for extracting clinical values
    PATTERNS = [
        # Dosing: 3-4 g/kg/day, 0.5-1 mg/kg/min
        r'(\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\d+(?:\.\d+)?)?)\s*(g|mg|mcg|μg|mEq|mmol|mL|L|IU|kcal|%)\s*/\s*(kg|day|hr|min|L)(?:\s*/\s*(day|hr|min))?',

        # Simple values with units: 150 mL, 2.5 g
        r'(\d+(?:\.\d+)?)\s*(g|mg|mcg|μg|mEq|mmol|mL|L|IU|kcal)\b',

        # Percentages: 20%, 10-20%
        r'(\d+(?:\.\d+)?(?:\s*(?:to|-)\s*\d+(?:\.\d+)?)?)\s*%',

        # Ranges without units: 3-4, 2.5 to 3.0
        r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(g|mg|mcg|mEq|mmol|mL|L|IU|kcal|%)?\s*(?:/\s*(kg|day|hr|min|L))?',
    ]

    def extract(self, text: str) -> List[ClinicalValue]:
        """Extract all clinical values from text."""
        values = []

        # Pattern 1: Full dosing (e.g., 3-4 g/kg/day)
        pattern1 = r'(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)\s*(g|mg|mcg|μg|mEq|mmol|mL|L|IU|kcal|%)\s*/\s*(kg|day|hr|min|L)(?:\s*/\s*(day|hr|min))?'
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            raw = match.group(0)
            val = match.group(1)
            unit = match.group(2)
            denom = match.group(3)
            if match.group(4):
                denom += "/" + match.group(4)
            values.append(ClinicalValue(
                raw=raw,
                value=val,
                unit=unit.lower(),
                denominator=denom.lower(),
                is_range="to" in val.lower() or "-" in val or "–" in val
            ))

        # Pattern 2: Simple values with units (e.g., 150 mL)
        pattern2 = r'(\d+(?:\.\d+)?)\s*(g|mg|mcg|μg|mEq|mmol|mL|L|IU|kcal)\b'
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            raw = match.group(0)
            # Skip if already captured by pattern 1
            if any(raw in v.raw for v in values):
                continue
            values.append(ClinicalValue(
                raw=raw,
                value=match.group(1),
                unit=match.group(2).lower(),
                is_range=False
            ))

        # Pattern 3: Percentages
        pattern3 = r'(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)\s*%'
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            raw = match.group(0)
            if any(raw in v.raw for v in values):
                continue
            values.append(ClinicalValue(
                raw=raw,
                value=match.group(1),
                unit="%",
                is_range="to" in match.group(1).lower() or "-" in match.group(1)
            ))

        return values

    def compare(self, expected_values: List[ClinicalValue], actual_values: List[ClinicalValue]) -> Dict:
        """Compare extracted values between expected and actual."""
        if not expected_values:
            return {
                "exact_matches": 0,
                "partial_matches": 0,
                "misses": 0,
                "extras": len(actual_values),
                "precision": 1.0 if not actual_values else 0.5,
                "recall": 1.0,
                "details": []
            }

        exact_matches = 0
        partial_matches = 0
        misses = 0
        details = []

        actual_normalized = {v.normalized(): v for v in actual_values}
        matched_actual = set()

        for exp in expected_values:
            exp_norm = exp.normalized()

            # Check for exact match
            if exp_norm in actual_normalized:
                exact_matches += 1
                matched_actual.add(exp_norm)
                details.append({"expected": exp.raw, "actual": exp.raw, "match": "exact"})
            else:
                # Check for partial match (same value, different format)
                found_partial = False
                for act_norm, act in actual_normalized.items():
                    if act_norm in matched_actual:
                        continue
                    # Same numeric value and unit?
                    if exp.value == act.value and exp.unit == act.unit:
                        partial_matches += 1
                        matched_actual.add(act_norm)
                        found_partial = True
                        details.append({"expected": exp.raw, "actual": act.raw, "match": "partial"})
                        break

                if not found_partial:
                    misses += 1
                    details.append({"expected": exp.raw, "actual": None, "match": "miss"})

        extras = len(actual_values) - len(matched_actual)

        precision = (exact_matches + 0.5 * partial_matches) / len(actual_values) if actual_values else 1.0
        recall = (exact_matches + 0.5 * partial_matches) / len(expected_values) if expected_values else 1.0

        return {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "misses": misses,
            "extras": extras,
            "precision": precision,
            "recall": recall,
            "details": details
        }


# ============================================================================
# SOURCE MATCHING
# ============================================================================

class SourceMatcher:
    """Match retrieved sources against expected sources."""

    ALIASES = {
        "aspen": ["aspen", "american society for parenteral", "a.s.p.e.n"],
        "nicu": ["nicu", "neonatal intensive care"],
        "pediatric": ["pediatric", "peds", "children"],
        "handbook": ["handbook", "manual", "guide"],
    }

    def normalize_source(self, source: str) -> str:
        """Normalize source name for comparison."""
        s = source.lower()
        s = re.sub(r'\.(md|json|pdf|txt)$', '', s)
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def sources_match(self, source1: str, source2: str, fuzzy: bool = True) -> bool:
        """Check if two sources match."""
        norm1 = self.normalize_source(source1)
        norm2 = self.normalize_source(source2)

        # Exact match
        if norm1 == norm2:
            return True

        # Substring match
        if norm1 in norm2 or norm2 in norm1:
            return True

        if fuzzy:
            # Word overlap
            words1 = set(norm1.split())
            words2 = set(norm2.split())
            overlap = len(words1 & words2)
            if overlap >= min(len(words1), len(words2)) * 0.5:
                return True

        return False

    def hit_at_k(self, retrieved_sources: List[str], expected_source: str, k: int) -> bool:
        """Check if expected source is in top-k retrieved."""
        for source in retrieved_sources[:k]:
            if self.sources_match(source, expected_source):
                return True
        return False

    def reciprocal_rank(self, retrieved_sources: List[str], expected_source: str) -> float:
        """Calculate reciprocal rank (1/rank of first match)."""
        for i, source in enumerate(retrieved_sources):
            if self.sources_match(source, expected_source):
                return 1.0 / (i + 1)
        return 0.0


# ============================================================================
# CITATION EXTRACTION
# ============================================================================

class CitationExtractor:
    """Extract and parse citations from text."""

    PATTERNS = [
        # [Source Name, p.XX]
        r'\[([^\]]+?),?\s*p\.?\s*(\d+)(?:[:\s][^\]]+)?\]',
        # According to Source (p.XX)
        r'According to (?:the )?([^(]+?)\s*\(p\.?\s*(\d+)\)',
        # Source, Page XX
        r'([^,\[\]]+?),\s*Page\s*(\d+)',
        # [Source: Name]
        r'\[Source:\s*([^\]]+)\]',
    ]

    def extract(self, text: str) -> List[Dict]:
        """Extract all citations from text."""
        citations = []

        for pattern in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    citations.append({
                        "source": match.group(1).strip(),
                        "page": int(match.group(2)) if match.group(2) else None,
                        "raw": match.group(0)
                    })
                elif len(match.groups()) == 1:
                    citations.append({
                        "source": match.group(1).strip(),
                        "page": None,
                        "raw": match.group(0)
                    })

        return citations


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Retrieval-specific metrics."""
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    mrr: float = 0.0  # Mean Reciprocal Rank
    expected_source: str = ""
    retrieved_sources: List[str] = field(default_factory=list)
    avg_similarity: float = 0.0


@dataclass
class ClinicalMetrics:
    """Clinical value comparison metrics."""
    value_precision: float = 0.0
    value_recall: float = 0.0
    exact_matches: int = 0
    partial_matches: int = 0
    misses: int = 0
    extras: int = 0
    expected_values: List[str] = field(default_factory=list)
    actual_values: List[str] = field(default_factory=list)
    comparison_details: List[Dict] = field(default_factory=list)


@dataclass
class CitationMetrics:
    """Citation quality metrics."""
    has_citations: bool = False
    num_citations: int = 0
    citations_grounded: int = 0  # Citations that match retrieved sources
    citations: List[Dict] = field(default_factory=list)


@dataclass
class SampleResult:
    """Complete evaluation result for one sample."""
    idx: int
    question: str
    expected_answer: str
    phase1_answer: str
    phase2_answer: str

    # Retrieval
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    # Phase 1 (no RAG)
    phase1_clinical: ClinicalMetrics = field(default_factory=ClinicalMetrics)
    phase1_citations: CitationMetrics = field(default_factory=CitationMetrics)

    # Phase 2 (with RAG)
    phase2_clinical: ClinicalMetrics = field(default_factory=ClinicalMetrics)
    phase2_citations: CitationMetrics = field(default_factory=CitationMetrics)

    # Diagnosis
    diagnosis: List[str] = field(default_factory=list)


# ============================================================================
# MODEL MANAGER (Reuse from enhanced_eval.py)
# ============================================================================

class ModelManager:
    """Singleton to manage model loading."""

    _instance = None
    _llm_model = None
    _llm_tokenizer = None
    _embed_model = None
    _reranker = None
    _chroma_collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_llm(self, model_name: str = "chandramax/tpn-gpt-oss-20b"):
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
            print("[ModelManager] LLM loaded")
        return self._llm_model, self._llm_tokenizer

    def load_embedding_model(self, model_name: str = None):
        # Use stored model name or default
        if model_name:
            self._embed_model_name = model_name
        elif not hasattr(self, '_embed_model_name') or self._embed_model_name is None:
            self._embed_model_name = "Qwen/Qwen3-Embedding-8B"

        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            print(f"[ModelManager] Loading embedding model: {self._embed_model_name}...")
            self._embed_model = SentenceTransformer(
                self._embed_model_name,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            print("[ModelManager] Embedding model loaded")
        return self._embed_model

    def set_embedding_model(self, model_name: str):
        """Set the embedding model to use (call before load_embedding_model)."""
        self._embed_model_name = model_name
        # Reset if already loaded with different model
        if self._embed_model is not None:
            del self._embed_model
            self._embed_model = None
            import torch
            torch.cuda.empty_cache()

    def load_reranker(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            print(f"[ModelManager] Loading reranker: {model_name}...")
            self._reranker = CrossEncoder(model_name, max_length=512)
            print("[ModelManager] Reranker loaded")
        return self._reranker

    def load_chroma(self):
        if self._chroma_collection is None:
            import chromadb

            chroma_path = project_root / "data" / "chroma"
            client = chromadb.PersistentClient(path=str(chroma_path))
            self._chroma_collection = client.get_collection("tpn_documents")
            print(f"[ModelManager] ChromaDB loaded: {self._chroma_collection.count()} docs")
        return self._chroma_collection

    def cleanup(self):
        import torch

        del self._llm_model, self._llm_tokenizer, self._embed_model, self._reranker
        self._llm_model = None
        self._llm_tokenizer = None
        self._embed_model = None
        self._reranker = None
        torch.cuda.empty_cache()


model_manager = ModelManager()


# ============================================================================
# RETRIEVAL WITH RERANKING
# ============================================================================

def retrieve_with_reranking(
    question: str,
    initial_k: int = 20,
    final_k: int = 5
) -> Tuple[str, List[str], List[Dict], float]:
    """
    Retrieve context with reranking for better precision.
    Returns: (formatted_context, context_list, sources, avg_score)
    """
    embed_model = model_manager.load_embedding_model()
    reranker = model_manager.load_reranker()
    collection = model_manager.load_chroma()

    # Initial retrieval
    query_embedding = embed_model.encode([question], prompt_name="query")[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"]
    )

    # Prepare for reranking
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

    # Rerank
    pairs = [(question, c['content']) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c['rerank_score'] = float(rerank_scores[i])

    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Take top_k
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
            'content_preview': c['content'][:200]
        })

    avg_score = sum(s['rerank_score'] for s in sources) / len(sources) if sources else 0

    return "\n\n---\n\n".join(context_parts), context_list, sources, avg_score


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def generate_answer(question: str, context: Optional[str] = None, max_tokens: int = 4096) -> str:
    """Generate answer with or without RAG context."""
    import torch

    model, tokenizer = model_manager.load_llm()

    if context:
        system_prompt = f"""You are a TPN clinical expert. Answer using the reference documents provided.

RULES:
- Be CONCISE (2-4 sentences for simple questions)
- Cite sources: [Source Name, p.XX]
- Include specific values with UNITS (g/kg/day, mEq/kg/day, etc.)
- Only elaborate if the question requires detailed explanation

REFERENCES:
{context}

Answer concisely with citations."""
    else:
        system_prompt = """You are a TPN clinical expert. Answer CONCISELY (2-4 sentences for simple questions). Include specific values with units."""

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract final answer from reasoning trace
    text = response.strip()
    if 'assistantfinal' in text.lower():
        idx = text.lower().rfind('assistantfinal')
        text = text[idx + len('assistantfinal'):].strip()

    return text


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def evaluate_sample(
    idx: int,
    question: str,
    expected_answer: str,
    expected_source: Optional[str] = None,
    verbose: bool = True
) -> SampleResult:
    """Run complete evaluation on one sample."""

    value_extractor = ClinicalValueExtractor()
    source_matcher = SourceMatcher()
    citation_extractor = CitationExtractor()

    result = SampleResult(
        idx=idx,
        question=question,
        expected_answer=expected_answer,
        phase1_answer="",
        phase2_answer=""
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"SAMPLE {idx + 1}")
        print(f"{'='*60}")
        print(f"Q: {question[:80]}...")

    # Extract expected clinical values
    expected_values = value_extractor.extract(expected_answer)

    # Phase 1: Model only
    if verbose:
        print("\n[Phase 1] Model only...")
    result.phase1_answer = generate_answer(question, context=None)

    # Evaluate Phase 1
    phase1_values = value_extractor.extract(result.phase1_answer)
    phase1_comparison = value_extractor.compare(expected_values, phase1_values)

    result.phase1_clinical = ClinicalMetrics(
        value_precision=phase1_comparison['precision'],
        value_recall=phase1_comparison['recall'],
        exact_matches=phase1_comparison['exact_matches'],
        partial_matches=phase1_comparison['partial_matches'],
        misses=phase1_comparison['misses'],
        extras=phase1_comparison['extras'],
        expected_values=[v.raw for v in expected_values],
        actual_values=[v.raw for v in phase1_values],
        comparison_details=phase1_comparison['details']
    )

    phase1_cits = citation_extractor.extract(result.phase1_answer)
    result.phase1_citations = CitationMetrics(
        has_citations=len(phase1_cits) > 0,
        num_citations=len(phase1_cits),
        citations=phase1_cits
    )

    # Retrieval
    if verbose:
        print("[Retrieval] Getting context...")
    formatted_context, context_list, sources, avg_score = retrieve_with_reranking(question)

    retrieved_sources = [s['source'] for s in sources]

    if expected_source:
        result.retrieval = RetrievalMetrics(
            hit_at_1=source_matcher.hit_at_k(retrieved_sources, expected_source, 1),
            hit_at_3=source_matcher.hit_at_k(retrieved_sources, expected_source, 3),
            hit_at_5=source_matcher.hit_at_k(retrieved_sources, expected_source, 5),
            mrr=source_matcher.reciprocal_rank(retrieved_sources, expected_source),
            expected_source=expected_source,
            retrieved_sources=retrieved_sources,
            avg_similarity=avg_score
        )
    else:
        result.retrieval = RetrievalMetrics(
            retrieved_sources=retrieved_sources,
            avg_similarity=avg_score
        )

    if verbose:
        print(f"  Retrieved: {retrieved_sources[:3]}...")
        print(f"  Avg rerank score: {avg_score:.3f}")

    # Phase 2: Model + RAG
    if verbose:
        print("[Phase 2] Model + RAG...")
    result.phase2_answer = generate_answer(question, context=formatted_context)

    # Evaluate Phase 2
    phase2_values = value_extractor.extract(result.phase2_answer)
    phase2_comparison = value_extractor.compare(expected_values, phase2_values)

    result.phase2_clinical = ClinicalMetrics(
        value_precision=phase2_comparison['precision'],
        value_recall=phase2_comparison['recall'],
        exact_matches=phase2_comparison['exact_matches'],
        partial_matches=phase2_comparison['partial_matches'],
        misses=phase2_comparison['misses'],
        extras=phase2_comparison['extras'],
        expected_values=[v.raw for v in expected_values],
        actual_values=[v.raw for v in phase2_values],
        comparison_details=phase2_comparison['details']
    )

    phase2_cits = citation_extractor.extract(result.phase2_answer)
    # Check if citations are grounded in retrieved sources
    grounded = 0
    for cit in phase2_cits:
        for src in retrieved_sources:
            if source_matcher.sources_match(cit['source'], src):
                grounded += 1
                break

    result.phase2_citations = CitationMetrics(
        has_citations=len(phase2_cits) > 0,
        num_citations=len(phase2_cits),
        citations_grounded=grounded,
        citations=phase2_cits
    )

    # Diagnosis
    diagnose_sample(result)

    if verbose:
        print_sample_result(result)

    return result


def diagnose_sample(result: SampleResult):
    """Add diagnostic insights to the result."""
    diag = []

    # Retrieval diagnosis
    if result.retrieval.expected_source:
        if not result.retrieval.hit_at_5:
            diag.append("RETRIEVAL_MISS: Expected source not in top 5")
        elif not result.retrieval.hit_at_1:
            diag.append("RETRIEVAL_LOW_RANK: Source found but not #1")

    if result.retrieval.avg_similarity < 0.3:
        diag.append("LOW_SIMILARITY: Retrieved docs have low relevance scores")

    # Clinical value diagnosis
    p1_recall = result.phase1_clinical.value_recall
    p2_recall = result.phase2_clinical.value_recall

    if p2_recall < p1_recall:
        diag.append("RAG_HURT: RAG made clinical values WORSE")
    elif p2_recall > p1_recall + 0.2:
        diag.append("RAG_HELPED: RAG significantly improved clinical values")
    elif p2_recall <= p1_recall + 0.05:
        diag.append("RAG_NO_EFFECT: RAG didn't improve clinical values")

    if result.phase2_clinical.misses > 0:
        diag.append(f"MISSING_VALUES: {result.phase2_clinical.misses} expected values not found")

    if result.phase2_clinical.extras > len(result.phase2_clinical.expected_values):
        diag.append("HALLUCINATED_VALUES: More values generated than expected")

    # Citation diagnosis
    if result.phase2_citations.has_citations:
        if result.phase2_citations.citations_grounded == 0:
            diag.append("UNGROUNDED_CITATIONS: Citations don't match retrieved sources")
        elif result.phase2_citations.citations_grounded < result.phase2_citations.num_citations:
            diag.append("PARTIAL_GROUNDING: Some citations not grounded")
    else:
        diag.append("NO_CITATIONS: Model didn't cite sources")

    result.diagnosis = diag


def print_sample_result(result: SampleResult):
    """Print formatted sample result."""
    print(f"\n{'-'*50}")
    print("CLINICAL VALUES:")
    print(f"  Expected: {result.phase2_clinical.expected_values[:5]}")
    print(f"  P1 Found: {result.phase1_clinical.actual_values[:5]}")
    print(f"  P2 Found: {result.phase2_clinical.actual_values[:5]}")

    print(f"\nMETRICS:")
    print(f"  {'Metric':<20} {'Phase1':>10} {'Phase2':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Value Recall':<20} {result.phase1_clinical.value_recall:>10.3f} {result.phase2_clinical.value_recall:>10.3f} {result.phase2_clinical.value_recall - result.phase1_clinical.value_recall:>+10.3f}")
    print(f"  {'Value Precision':<20} {result.phase1_clinical.value_precision:>10.3f} {result.phase2_clinical.value_precision:>10.3f} {result.phase2_clinical.value_precision - result.phase1_clinical.value_precision:>+10.3f}")
    print(f"  {'Exact Matches':<20} {result.phase1_clinical.exact_matches:>10} {result.phase2_clinical.exact_matches:>10}")
    print(f"  {'Citations':<20} {result.phase1_citations.num_citations:>10} {result.phase2_citations.num_citations:>10}")

    if result.retrieval.expected_source:
        print(f"\nRETRIEVAL:")
        print(f"  Expected: {result.retrieval.expected_source}")
        print(f"  Hit@1: {result.retrieval.hit_at_1}, Hit@5: {result.retrieval.hit_at_5}, MRR: {result.retrieval.mrr:.3f}")

    if result.diagnosis:
        print(f"\nDIAGNOSIS:")
        for d in result.diagnosis:
            print(f"  [{d}]")


# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

def print_aggregate_results(results: List[SampleResult]):
    """Print aggregate statistics."""
    n = len(results)
    if n == 0:
        return

    # Aggregate metrics
    p1_recall = sum(r.phase1_clinical.value_recall for r in results) / n
    p2_recall = sum(r.phase2_clinical.value_recall for r in results) / n
    p1_precision = sum(r.phase1_clinical.value_precision for r in results) / n
    p2_precision = sum(r.phase2_clinical.value_precision for r in results) / n

    p1_exact = sum(r.phase1_clinical.exact_matches for r in results)
    p2_exact = sum(r.phase2_clinical.exact_matches for r in results)

    # Retrieval
    hit1 = sum(1 for r in results if r.retrieval.hit_at_1) / n if any(r.retrieval.expected_source for r in results) else 0
    hit5 = sum(1 for r in results if r.retrieval.hit_at_5) / n if any(r.retrieval.expected_source for r in results) else 0
    avg_mrr = sum(r.retrieval.mrr for r in results) / n if any(r.retrieval.expected_source for r in results) else 0

    # Diagnosis counts
    diag_counts = Counter()
    for r in results:
        for d in r.diagnosis:
            key = d.split(":")[0]
            diag_counts[key] += 1

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({n} samples)")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Phase1':>10} {'Phase2':>10} {'Delta':>10} {'RAG Lift':>10}")
    print("-" * 65)
    print(f"{'Clinical Value Recall':<25} {p1_recall:>10.3f} {p2_recall:>10.3f} {p2_recall - p1_recall:>+10.3f} {((p2_recall - p1_recall) / p1_recall * 100) if p1_recall > 0 else 0:>+9.1f}%")
    print(f"{'Clinical Value Precision':<25} {p1_precision:>10.3f} {p2_precision:>10.3f} {p2_precision - p1_precision:>+10.3f} {((p2_precision - p1_precision) / p1_precision * 100) if p1_precision > 0 else 0:>+9.1f}%")
    print(f"{'Total Exact Matches':<25} {p1_exact:>10} {p2_exact:>10} {p2_exact - p1_exact:>+10}")

    print(f"\nRETRIEVAL QUALITY:")
    print("-" * 40)
    print(f"  Hit@1: {hit1:.1%}")
    print(f"  Hit@5: {hit5:.1%}")
    print(f"  MRR:   {avg_mrr:.3f}")

    print(f"\nDIAGNOSIS SUMMARY:")
    print("-" * 40)
    for diag, count in diag_counts.most_common(10):
        pct = count / n * 100
        print(f"  {diag:<30} {count:>5} ({pct:>5.1f}%)")

    print("\n" + "=" * 70)
    print("ACTIONABLE INSIGHTS:")
    print("=" * 70)

    if diag_counts.get("RETRIEVAL_MISS", 0) > n * 0.3:
        print("  [!] HIGH RETRIEVAL MISS RATE")
        print("      → Improve chunking or add document coverage")

    if diag_counts.get("RAG_NO_EFFECT", 0) > n * 0.3:
        print("  [!] RAG NOT HELPING")
        print("      → Model may be ignoring context; improve prompt")

    if diag_counts.get("UNGROUNDED_CITATIONS", 0) > n * 0.3:
        print("  [!] CITATION HALLUCINATION")
        print("      → Add citation grounding post-processing")

    if diag_counts.get("MISSING_VALUES", 0) > n * 0.3:
        print("  [!] MISSING CLINICAL VALUES")
        print("      → Check if retrieval is finding the right content")

    if p2_recall >= 0.9:
        print("  [✓] TARGET ACHIEVED: Clinical value recall >= 90%!")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_samples(test_file: Path, num_samples: int) -> List[Dict]:
    """Load test samples from JSONL."""
    samples = []

    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            data = json.loads(line)
            question = None
            expected = None

            for msg in data['messages']:
                if msg['role'] == 'user':
                    question = msg['content']
                elif msg['role'] == 'assistant':
                    expected = msg.get('content', '')

            # Try to extract source citation from expected answer
            source = None
            source_match = re.search(r'According to (?:the )?([^(]+?)\s*\(p\.', expected or "")
            if source_match:
                source = source_match.group(1).strip()

            samples.append({
                'idx': i,
                'question': question,
                'expected': expected,
                'source': source
            })

    return samples


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Available embedding models
    EMBEDDING_MODELS = [
        ("1", "Qwen/Qwen3-Embedding-8B", "General purpose (current)"),
        ("2", "abhinand/MedEmbed-large-v0.1", "Medical IR - RECOMMENDED for TPN"),
        ("3", "abhinand/MedEmbed-base-v0.1", "Medical IR - faster"),
        ("4", "tencent/KaLM-Embedding-Gemma3-12B-2511", "Best MMTEB - larger"),
        ("5", "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5", "SOTA compact"),
    ]

    parser = argparse.ArgumentParser(description='Clinical TPN RAG Evaluation')
    parser.add_argument('--samples', '-n', type=int, default=5, help='Number of samples')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less verbose')
    parser.add_argument('--diagnose', '-d', action='store_true', help='Show diagnosis details')
    parser.add_argument('--embed-model', '-e', type=str, default=None,
                        help='Embedding model (1-5 or full HF ID)')
    args = parser.parse_args()

    test_file = project_root / "eval" / "data" / "test_with_citations.jsonl"
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)

    print("=" * 70)
    print("CLINICAL TPN RAG EVALUATION")
    print("=" * 70)
    print("This evaluation focuses on:")
    print("  1. EXACT clinical value matching (doses, units, ranges)")
    print("  2. Retrieval quality (Hit@K, source matching)")
    print("  3. Citation grounding (are citations real?)")
    print("  4. Actionable diagnostics")
    print("=" * 70)

    # Select embedding model
    if args.embed_model:
        if args.embed_model in [m[0] for m in EMBEDDING_MODELS]:
            embed_model = next(m[1] for m in EMBEDDING_MODELS if m[0] == args.embed_model)
        else:
            embed_model = args.embed_model
    else:
        print("\n📊 SELECT EMBEDDING MODEL:")
        print("-" * 50)
        for num, model_id, desc in EMBEDDING_MODELS:
            marker = "⭐" if "RECOMMENDED" in desc else "  "
            print(f"  {marker} {num}. {model_id}")
            print(f"       {desc}")
        print("-" * 50)
        choice = input("Select (1-5, default=1): ").strip() or "1"
        if choice in [m[0] for m in EMBEDDING_MODELS]:
            embed_model = next(m[1] for m in EMBEDDING_MODELS if m[0] == choice)
        else:
            embed_model = EMBEDDING_MODELS[0][1]

    print(f"\n✅ Using embedding model: {embed_model}")

    # Set the embedding model in ModelManager
    model_manager.set_embedding_model(embed_model)

    # Load samples
    samples = load_samples(test_file, args.samples)
    print(f"\nLoaded {len(samples)} samples")

    # Run evaluation
    results = []
    for sample in samples:
        try:
            result = evaluate_sample(
                idx=sample['idx'],
                question=sample['question'],
                expected_answer=sample['expected'],
                expected_source=sample['source'],
                verbose=not args.quiet
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR on sample {sample['idx']}: {e}")
            import traceback
            traceback.print_exc()

    # Print aggregate
    print_aggregate_results(results)

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "eval" / f"clinical_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'embedding_model': embed_model,
            'num_samples': len(results),
            'results': [asdict(r) for r in results]
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Cleanup
    model_manager.cleanup()


if __name__ == "__main__":
    main()
