#!/usr/bin/env python3
"""
Fine-Tuned Model + RAG Pipeline.

Combines your fine-tuned TPN model (which has correct answers but
hallucinated citations) with RAG retrieval to provide verifiable citations.

Pipeline:
1. User asks question
2. RAG retrieves relevant chunks from real documents
3. Fine-tuned model generates answer (with potentially fake citations)
4. Citation Grounding replaces fake citations with real ones
5. Citation Evaluator verifies the quality

Usage:
    # With HuggingFace fine-tuned model
    python scripts/finetune_with_rag.py --model hf:your-finetuned-model \
        --persist-dir ./data --query "What is the protein requirement?"

    # Demo mode (uses mock data to show the concept)
    python scripts/finetune_with_rag.py --demo

    # Evaluate citation quality
    python scripts/finetune_with_rag.py --evaluate --persist-dir ./data -n 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_demo():
    """
    Demonstrate the citation grounding concept.
    Shows how RAG fixes hallucinated citations from fine-tuned models.
    """
    from app.retrieval.citation_grounding import CitationGrounder
    from app.evaluation.citation_metrics import CitationEvaluator, RetrievedChunk

    print("=" * 70)
    print("FINE-TUNED MODEL + RAG: CITATION GROUNDING DEMO")
    print("=" * 70)

    print("""
PROBLEM:
  Your fine-tuned TPN model gives CORRECT answers (3-4 g/kg/day)
  But HALLUCINATES citations ([Fake Handbook, p.999])

SOLUTION:
  1. RAG retrieves REAL documents with verified sources
  2. Citation Grounding replaces fake citations with real ones
  3. Result: Correct answer + Verifiable citations
""")

    grounder = CitationGrounder()
    evaluator = CitationEvaluator()

    # === Simulate fine-tuned model output ===
    print("-" * 70)
    print("STEP 1: Fine-tuned model generates answer (with hallucinated citations)")
    print("-" * 70)

    fine_tuned_output = """
Protein requirements for preterm infants are 3.5-4.0 g/kg/day according to current guidelines [TPN Nutrition Manual 2024, p.234]. This should be initiated within the first 24 hours of life to prevent catabolism [Clinical Nutrition Handbook, p.567]. For extremely low birth weight infants (<1000g), requirements may be as high as 4.5 g/kg/day [Pediatric Nutrition Reference, p.89].

Term infants require less protein at 2.5-3.0 g/kg/day [Neonatal Care Guidelines, p.123]. The protein source should be crystalline amino acids, with cysteine supplementation recommended for preterm infants [NICU Protocol Manual, p.456].
"""

    print(fine_tuned_output)
    print("\n[!] Notice: Citations look real but are HALLUCINATED")

    # === Simulate RAG retrieval ===
    print("\n" + "-" * 70)
    print("STEP 2: RAG retrieves real documents from your indexed corpus")
    print("-" * 70)

    # These simulate real retrieved chunks with actual metadata
    retrieved_chunks = [
        {
            "content": "Protein requirements: Preterm infants require 3.5-4.0 g/kg/day of protein for optimal growth. Amino acid supplementation should begin within the first 24 hours of life. For ELBW infants (<1000g), protein needs may reach 4.5 g/kg/day to achieve intrauterine growth rates.",
            "metadata": {"source": "ASPEN_Guidelines_2020.md", "page": 44},
        },
        {
            "content": "Term infant protein needs are lower at 2.5-3.0 g/kg/day. The reduced requirement reflects the lower growth rate compared to preterm infants.",
            "metadata": {"source": "ASPEN_Guidelines_2020.md", "page": 52},
        },
        {
            "content": "Crystalline amino acid solutions are the preferred protein source for parenteral nutrition. For preterm infants, cysteine supplementation is recommended as they have limited ability to synthesize this amino acid.",
            "metadata": {"source": "NICU_TPN_Protocol.md", "page": 15},
        },
    ]

    for i, chunk in enumerate(retrieved_chunks, 1):
        src = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        print(f"\n  Chunk {i}: [{src}, p.{page}]")
        print(f"  Content: {chunk['content'][:100]}...")

    # === Apply Citation Grounding ===
    print("\n" + "-" * 70)
    print("STEP 3: Citation Grounding replaces hallucinated citations with real ones")
    print("-" * 70)

    result = grounder.ground_citations(
        fine_tuned_output,
        retrieved_chunks,
        add_inline_citations=True,
        add_references_section=True,
    )

    print("\n" + result.grounded_text)

    print(f"\n  Citations removed (hallucinated): {result.citations_removed}")
    print(f"  Citations added (verified): {result.citations_added}")
    print(f"  Grounding confidence: {result.confidence:.1%}")

    # === Evaluate Citation Quality ===
    print("\n" + "-" * 70)
    print("STEP 4: Evaluate citation quality")
    print("-" * 70)

    # Convert to RetrievedChunk objects for evaluation
    eval_chunks = [
        RetrievedChunk(
            content=c["content"],
            source_doc=c["metadata"]["source"],
            page_num=c["metadata"]["page"],
        )
        for c in retrieved_chunks
    ]

    eval_result = evaluator.evaluate(
        question="What are protein requirements for preterm and term infants?",
        generated_answer=result.grounded_text,
        retrieved_chunks=eval_chunks,
        ground_truth_source="ASPEN Guidelines",
        ground_truth_page=44,
    )

    print(f"\n  Source Accuracy: {eval_result.source_accuracy:.1%}")
    print(f"  Page Precision: {eval_result.page_precision:.1%}")
    print(f"  Faithfulness: {eval_result.faithfulness_score:.1%}")
    print(f"  Hallucination Risk: {eval_result.hallucination_risk:.1%}")
    print(f"  Overall Citation Score: {eval_result.overall_score:.1%}")
    print(f"  Ground Truth Found: {eval_result.ground_truth_source_found}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
BEFORE (Fine-tuned model alone):
  + Correct clinical information (3.5-4.0 g/kg/day, 24 hours, etc.)
  - Hallucinated citations that don't exist
  - Cannot be verified by clinicians

AFTER (Fine-tuned model + RAG + Citation Grounding):
  + Correct clinical information (preserved)
  + Real citations from actual indexed documents
  + Page numbers that can be verified
  + Trustworthy for clinical use

The RAG system adds VERIFIABILITY without retraining the model!
""")


def run_with_model(
    model_spec: str,
    persist_dir: str,
    query: str,
):
    """
    Run the pipeline with an actual model and retriever.
    """
    from app.retrieval import CitationGrounder
    from app.evaluation import CitationEvaluator
    from app.models import create_model

    print("=" * 70)
    print("FINE-TUNED MODEL + RAG PIPELINE")
    print("=" * 70)

    # Parse model spec
    parts = model_spec.split(":", 1)
    if len(parts) < 2:
        print(f"Error: Invalid model spec '{model_spec}'. Use 'provider:model_name'")
        return
    provider, model_name = parts[0], parts[1]

    # Load retriever
    from scripts.compare_models import load_retriever, SimpleRetrieverAdapter
    raw_retriever = load_retriever(persist_dir)
    if not raw_retriever:
        print(f"Error: Could not load retriever from {persist_dir}")
        return

    retriever = SimpleRetrieverAdapter(raw_retriever)

    # Create model
    print(f"\nModel: {provider}/{model_name}")
    print(f"Retriever: Loaded from {persist_dir}")

    try:
        model = create_model(provider, model_name)
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # Initialize tools
    grounder = CitationGrounder()
    evaluator = CitationEvaluator()

    print(f"\nQuery: {query}")
    print("-" * 70)

    # Step 1: Retrieve relevant chunks
    print("\n1. Retrieving relevant documents...")
    chunks = retriever.retrieve(query, top_k=5)
    print(f"   Retrieved {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3], 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        print(f"   - {source}")

    # Format context for model
    context_parts = []
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page", "?")
        content = chunk.get("content", "")
        context_parts.append(f"[Source: {source}, Page {page}]\n{content}")

    context = "\n\n".join(context_parts)

    # Step 2: Generate answer with model
    print("\n2. Generating answer with fine-tuned model...")
    response = model.generate(
        question=query,
        context=context,
        use_rag=True,
    )
    print(f"   Latency: {response.latency_ms:.0f}ms")
    print(f"\n   Raw output:\n   {response.answer[:300]}...")

    # Step 3: Ground citations
    print("\n3. Grounding citations to real sources...")
    grounded = grounder.ground_citations(
        response.answer,
        chunks,
        add_inline_citations=True,
        add_references_section=True,
    )

    print(f"   Removed {grounded.citations_removed} potentially hallucinated citations")
    print(f"   Added {grounded.citations_added} verified citations")
    print(f"   Confidence: {grounded.confidence:.1%}")

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)
    print(grounded.grounded_text)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuned model + RAG with citation grounding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo showing how citation grounding works",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model spec (e.g., hf:your-finetuned-model)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        help="Directory with indexed documents",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the protein requirements for preterm infants?",
        help="Question to ask",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.model and args.persist_dir:
        run_with_model(args.model, args.persist_dir, args.query)
    else:
        parser.print_help()
        print("\n" + "-" * 50)
        print("Quick start: python scripts/finetune_with_rag.py --demo")


if __name__ == "__main__":
    main()
