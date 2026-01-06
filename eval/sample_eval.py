"""
Sample-Based Two-Phase Evaluation Script.

This script tests the evaluation pipeline on a single sample before scaling up.
Following iterative approach: analyze 1 sample, verify results, then scale.

Phase 1: Fine-tuned model ONLY (no RAG context)
Phase 2: Fine-tuned model + RAG (with retrieved context and citations)

Usage:
    python eval/sample_eval.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def load_single_sample(test_file: str, sample_idx: int = 0) -> dict:
    """Load a single sample from the JSONL test file."""
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise IndexError(f"Sample index {sample_idx} not found in {test_file}")


def extract_qa_from_sample(sample: dict) -> tuple:
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


def run_phase1_model_only(question: str, model_name: str = "chandramax/tpn-gpt-oss-20b") -> str:
    """
    Phase 1: Run fine-tuned model WITHOUT RAG context.
    This establishes baseline - what the model knows on its own.
    """
    print("\n" + "="*70)
    print("PHASE 1: Fine-tuned Model ONLY (No RAG)")
    print("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # System prompt matching fine-tuning format
    system_prompt = """You are a clinical expert specializing in neonatal and pediatric Total Parenteral Nutrition (TPN). Provide accurate, evidence-based guidance for TPN management including dosing calculations, monitoring protocols, and complication management. Always show your reasoning step-by-step."""

    # Format as chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response (no RAG context)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Cleanup model to free memory
    del model
    torch.cuda.empty_cache()

    return response.strip()


def run_phase2_model_with_rag(question: str, model_name: str = "chandramax/tpn-gpt-oss-20b") -> tuple:
    """
    Phase 2: Run fine-tuned model WITH RAG context.
    Retrieves relevant documents and grounds citations.
    """
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuned Model + RAG")
    print("="*70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # First, retrieve context using RAG
    print("Retrieving relevant documents...")
    retrieved_context, sources = retrieve_context(question)

    print(f"Retrieved {len(sources)} sources:")
    for i, src in enumerate(sources[:5], 1):
        print(f"  [{i}] {src['source']} (score: {src['score']:.3f})")

    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Hospital-Grade RAG Prompt
    system_prompt = """<role>
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

    formatted_system = system_prompt.format(context=retrieved_context)

    # Use "developer" role to match fine-tuning format
    messages = [
        {"role": "developer", "content": formatted_system},
        {"role": "user", "content": question}
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|system|>\n{formatted_system}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response (with RAG context)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()

    return response.strip(), sources


def retrieve_context(question: str, top_k: int = 5) -> tuple:
    """Retrieve relevant context from ChromaDB."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    import torch

    # Load embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    # Connect to ChromaDB
    chroma_path = project_root / "data" / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        collection = client.get_collection("tpn_documents")
    except Exception as e:
        print(f"Error: Could not find ChromaDB collection: {e}")
        return "", []

    # Embed query
    query_embedding = embed_model.encode([question], prompt_name="query")[0].tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Build context
    context_parts = []
    sources = []

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        distance = results['distances'][0][i] if results['distances'] else 0

        source_name = meta.get('source', meta.get('document_name', 'Unknown'))
        page_num = meta.get('page_num', '')
        section = meta.get('section', 'General')

        # Convert distance to similarity score
        score = 1 - distance  # cosine distance to similarity

        source_ref = f"{source_name}"
        if page_num:
            source_ref += f", Page {page_num}"

        context_parts.append(f"[Source: {source_ref}]\nSection: {section}\n{doc}")
        sources.append({
            'source': source_name,
            'page': page_num,
            'section': section,
            'score': score,
            'content': doc[:200] + "..."
        })

    del embed_model
    torch.cuda.empty_cache()

    return "\n\n---\n\n".join(context_parts), sources


def run_geval_comparison(question: str, expected: str, phase1_answer: str, phase2_answer: str):
    """
    Run GEval metrics on both phase answers.
    Uses GPT-4o-mini as judge for clinical correctness.
    """
    print("\n" + "="*70)
    print("GEVAL COMPARISON")
    print("="*70)

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    # Clinical correctness metric
    clinical_metric = GEval(
        name="Clinical Correctness",
        criteria="""Evaluate whether the response provides clinically correct information for TPN (Total Parenteral Nutrition) management. Consider:
1. Accuracy of dosing values, ranges, and units (e.g., mg/kg/day, g/kg/day, mL/kg/day)
2. Correctness of clinical protocols and monitoring recommendations
3. Accuracy of medication/nutrient interactions and contraindications
4. Proper clinical reasoning and evidence-based approach""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Compare dosing values and ranges in the response against the expected answer",
            "Verify clinical protocols and recommendations match established guidelines",
            "Check if the clinical reasoning is sound and evidence-based",
            "Assess whether any critical safety information is missing or incorrect"
        ],
        model="gpt-4o-mini",
        threshold=0.7
    )

    # Citation quality metric
    citation_metric = GEval(
        name="Citation Quality",
        criteria="""Evaluate the quality and accuracy of citations in the response:
1. Are sources properly cited with document names and page numbers?
2. Do the citations appear to reference real clinical guidelines (ASPEN, etc.)?
3. Are quoted passages properly attributed?
4. Is the citation format consistent?""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Identify all citations in the response",
            "Check if citations include document name and page reference",
            "Verify citation format is consistent",
            "Assess if citations appear relevant to the claims made"
        ],
        model="gpt-4o-mini",
        threshold=0.5
    )

    print("\n--- Phase 1 Evaluation (Model Only) ---")
    test_case_p1 = LLMTestCase(
        input=question,
        actual_output=phase1_answer,
        expected_output=expected
    )

    try:
        clinical_metric.measure(test_case_p1)
        print(f"Clinical Correctness: {clinical_metric.score:.3f}")
        print(f"  Reason: {clinical_metric.reason[:200]}...")

        citation_metric.measure(test_case_p1)
        print(f"Citation Quality: {citation_metric.score:.3f}")
        print(f"  Reason: {citation_metric.reason[:200]}...")

        p1_clinical = clinical_metric.score
        p1_citation = citation_metric.score
    except Exception as e:
        print(f"Error in Phase 1 evaluation: {e}")
        p1_clinical = 0
        p1_citation = 0

    print("\n--- Phase 2 Evaluation (Model + RAG) ---")
    test_case_p2 = LLMTestCase(
        input=question,
        actual_output=phase2_answer,
        expected_output=expected
    )

    try:
        clinical_metric.measure(test_case_p2)
        print(f"Clinical Correctness: {clinical_metric.score:.3f}")
        print(f"  Reason: {clinical_metric.reason[:200]}...")

        citation_metric.measure(test_case_p2)
        print(f"Citation Quality: {citation_metric.score:.3f}")
        print(f"  Reason: {citation_metric.reason[:200]}...")

        p2_clinical = clinical_metric.score
        p2_citation = citation_metric.score
    except Exception as e:
        print(f"Error in Phase 2 evaluation: {e}")
        p2_clinical = 0
        p2_citation = 0

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"                      Phase 1 (No RAG)    Phase 2 (With RAG)    Delta")
    print(f"Clinical Correctness:      {p1_clinical:.3f}              {p2_clinical:.3f}           {p2_clinical - p1_clinical:+.3f}")
    print(f"Citation Quality:          {p1_citation:.3f}              {p2_citation:.3f}           {p2_citation - p1_citation:+.3f}")

    return {
        'phase1': {'clinical': p1_clinical, 'citation': p1_citation},
        'phase2': {'clinical': p2_clinical, 'citation': p2_citation}
    }


def main():
    """Main evaluation flow."""
    print("="*70)
    print("SAMPLE-BASED TWO-PHASE EVALUATION")
    print("Testing on 1 sample before scaling up")
    print("="*70)

    # Check for required env vars
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Required for GEval.")
        print("Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    # Load test sample
    test_file = project_root / "eval" / "data" / "test_with_citations.jsonl"
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)

    print(f"\nLoading sample from: {test_file}")
    sample = load_single_sample(str(test_file), sample_idx=0)

    question, expected_answer, expected_thinking = extract_qa_from_sample(sample)

    print("\n" + "-"*70)
    print("SAMPLE QUESTION:")
    print("-"*70)
    print(question)

    print("\n" + "-"*70)
    print("EXPECTED ANSWER:")
    print("-"*70)
    print(expected_answer[:500] + "..." if len(expected_answer) > 500 else expected_answer)

    # Run Phase 1
    print("\n\nStarting Phase 1...")
    phase1_answer = run_phase1_model_only(question)

    print("\n" + "-"*70)
    print("PHASE 1 OUTPUT (Model Only):")
    print("-"*70)
    print(phase1_answer[:800] if len(phase1_answer) > 800 else phase1_answer)

    # Run Phase 2
    print("\n\nStarting Phase 2...")
    phase2_answer, sources = run_phase2_model_with_rag(question)

    print("\n" + "-"*70)
    print("PHASE 2 OUTPUT (Model + RAG):")
    print("-"*70)
    print(phase2_answer[:800] if len(phase2_answer) > 800 else phase2_answer)

    # Run GEval comparison
    print("\n\nRunning GEval comparison...")
    scores = run_geval_comparison(question, expected_answer, phase1_answer, phase2_answer)

    # Save results
    results = {
        'question': question,
        'expected_answer': expected_answer,
        'phase1_answer': phase1_answer,
        'phase2_answer': phase2_answer,
        'sources': sources,
        'scores': scores
    }

    output_file = project_root / "eval" / "sample_result.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
    print("\nIf satisfied with sample results, scale up with full evaluation.")


if __name__ == "__main__":
    main()
