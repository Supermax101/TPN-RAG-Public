"""
RAGAS Evaluation - Industry-Standard RAG Metrics
Uses OpenAI as the judge LLM for evaluation.

Metrics:
- Faithfulness: Is answer grounded in retrieved context?
- Answer Relevancy: Is answer relevant to the question?
- Context Precision: Are relevant docs ranked higher?
- Context Recall: Are all needed docs retrieved?
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import json

# Load .env file first
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.providers.embeddings import HuggingFaceEmbeddingProvider
from app.providers.vectorstore import ChromaVectorStore
from app.models import HuggingFaceProvider
from app.services.rag import RAGService
from app.services.hybrid_rag import HybridRAGService
from app.services.advanced_rag import AdvancedRAGConfig
from app.models import SearchQuery


def check_openai_key():
    """Check if OpenAI API key is configured and set it in environment."""
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found.")
        print("RAGAS requires OpenAI for evaluation metrics.")
        print("\nSet it in .env file:")
        print("  OPENAI_API_KEY=sk-...")
        return False
    
    # Explicitly set in environment for RAGAS to pick up
    os.environ["OPENAI_API_KEY"] = api_key
    return True


async def prepare_ragas_dataset(
    csv_path: str,
    rag_service: RAGService,
    max_questions: int = None
) -> List[Dict[str, Any]]:
    """Prepare dataset in RAGAS format by running RAG on each question."""
    
    df = pd.read_csv(csv_path, keep_default_na=False)
    
    if max_questions:
        df = df.head(max_questions)
    
    print(f"Preparing RAGAS dataset for {len(df)} questions...")
    
    dataset = []
    
    for idx, row in df.iterrows():
        question = row['question']
        correct_answer = row['correct_answer']
        case_context = row.get('case_context', '')
        
        # Build search query
        search_text = f"{case_context} {question}".strip() if case_context else question
        search_query = SearchQuery(query=search_text, limit=5)
        
        # Get search results
        search_response = await rag_service.search(search_query)
        results = search_response.results if hasattr(search_response, 'results') else search_response
        
        # Extract contexts
        contexts = []
        for r in results:
            if hasattr(r, 'chunk'):
                contexts.append(r.chunk.content)
            elif isinstance(r, dict):
                contexts.append(r.get('content', ''))
        
        # Generate answer
        context_text = "\n\n".join(contexts) if contexts else "No context found."
        prompt = f"""Answer this TPN question based on the context.

Question: {question}
Options: {row['options']}

Context:
{context_text}

Answer with just the letter (A, B, C, D, E, or F):"""
        
        answer = await rag_service.llm_provider.generate(prompt)
        
        # Clean answer - extract first letter
        import re
        match = re.search(r'\b([A-F])\b', answer.upper())
        generated_answer = match.group(1) if match else answer[:50]
        
        dataset.append({
            "question": question,
            "answer": generated_answer,
            "contexts": contexts,
            "ground_truth": correct_answer
        })
        
        print(f"  {idx + 1}/{len(df)}: Q{row['question_id']} - Generated: {generated_answer}, Expected: {correct_answer}")
    
    return dataset


def run_ragas_evaluation(dataset: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, float]:
    """Run RAGAS evaluation on the dataset."""
    
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        print("\nERROR: RAGAS not installed. Run:")
        print("  uv add ragas datasets langchain-openai")
        return {}
    
    # Configure LLM for RAGAS
    llm = LangchainLLMWrapper(ChatOpenAI(model=model))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Convert to HuggingFace Dataset format
    hf_dataset = Dataset.from_list(dataset)
    
    print(f"\nRunning RAGAS evaluation with {model}...")
    print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    
    # Run evaluation
    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    
    return result


def print_ragas_results(results, dataset: List[Dict[str, Any]]):
    """Print RAGAS evaluation results."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    
    console = Console()
    
    print("\n" + "=" * 75)
    print("                    RAGAS EVALUATION RESULTS")
    print("                  (Industry-Standard RAG Metrics)")
    print("=" * 75)
    
    # Convert RAGAS result to dictionary
    try:
        # RAGAS returns an EvaluationResult object - convert to dict
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            scores = df.mean().to_dict()
        elif hasattr(results, '_scores_dict'):
            scores = results._scores_dict
        elif isinstance(results, dict):
            scores = results
        else:
            # Try to iterate and get mean scores
            scores = {}
            for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                try:
                    scores[col] = float(results[col]) if col in results else None
                except:
                    pass
    except Exception as e:
        print(f"Error parsing results: {e}")
        print(f"Raw results: {results}")
        return
    
    # Metrics table
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Metric", width=25)
    table.add_column("Score", justify="right", width=12)
    table.add_column("Interpretation", width=30)
    
    metric_info = {
        "faithfulness": "Answer grounded in context?",
        "answer_relevancy": "Answer relevant to question?",
        "context_precision": "Relevant docs ranked higher?",
        "context_recall": "All needed docs retrieved?",
    }
    
    for metric, description in metric_info.items():
        if metric in scores and scores[metric] is not None:
            score = float(scores[metric])
            grade = "Excellent" if score >= 0.8 else "Good" if score >= 0.6 else "Needs Improvement"
            table.add_row(metric.replace("_", " ").title(), f"{score:.3f}", f"{grade} - {description}")
    
    console.print(table)
    
    # Calculate accuracy
    correct = sum(1 for d in dataset if d['answer'].upper() == d['ground_truth'].upper())
    accuracy = correct / len(dataset) * 100
    
    print(f"\nAccuracy: {correct}/{len(dataset)} = {accuracy:.1f}%")
    print("=" * 75)


async def main():
    """Main RAGAS evaluation function."""
    
    print("\n" + "=" * 75)
    print("RAGAS EVALUATION - Industry-Standard RAG Metrics")
    print("=" * 75)
    
    # Check OpenAI key
    if not check_openai_key():
        return
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    # Select RAG mode
    print("\nRAG Mode:")
    print("  1. Simple RAG")
    print("  2. Advanced RAG (BM25 + Cross-Encoder)")
    
    mode = input("\nSelect (1 or 2, default=1): ").strip()
    use_advanced = mode == "2"
    
    # Question limit
    limit_input = input("Limit questions? (Enter for all, or number): ").strip()
    max_questions = int(limit_input) if limit_input.isdigit() else None
    
    # Select OpenAI model for RAGAS evaluation
    print("\nOpenAI Model for RAGAS (judge model):")
    print("  1. gpt-5-mini (recommended)")
    print("  2. gpt-4o (high quality)")
    print("  3. gpt-4o-mini (legacy)")

    model_choice = input("\nSelect (1-3, default=1): ").strip()
    model_map = {"1": "gpt-5-mini", "2": "gpt-4o", "3": "gpt-4o-mini"}
    ragas_model = model_map.get(model_choice, "gpt-5-mini")
    print(f"RAGAS judge model: {ragas_model}")

    # Select HuggingFace LLM model for answer generation
    print("\nHuggingFace LLM Model (for answer generation):")
    hf_models = [
        "chandramax/tpn-gpt-oss-20b",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    for i, model in enumerate(hf_models, 1):
        print(f"  {i}. {model}")

    hf_choice = input(f"\nSelect (1-{len(hf_models)}, default=1): ").strip()
    if hf_choice.isdigit() and 1 <= int(hf_choice) <= len(hf_models):
        selected_hf_model = hf_models[int(hf_choice) - 1]
    else:
        selected_hf_model = hf_models[0]
    print(f"Using HuggingFace model: {selected_hf_model}")

    # Initialize RAG service
    print("\nInitializing RAG service...")
    embedding_provider = HuggingFaceEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = HuggingFaceProvider(model_name=selected_hf_model)
    
    if use_advanced:
        config = AdvancedRAGConfig(
            enable_bm25_hybrid=True,
            enable_cross_encoder=True,
        )
        rag_service = HybridRAGService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            advanced_config=config
        )
        print("Mode: Advanced RAG")
    else:
        rag_service = RAGService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider
        )
        print("Mode: Simple RAG")
    
    # Verify collection
    stats = await rag_service.get_collection_stats()
    if stats.get("total_chunks", 0) == 0:
        print("ERROR: No documents in collection. Run 'init' first.")
        return
    
    print(f"Collection: {stats.get('total_chunks')} chunks")
    
    # Prepare dataset
    dataset = await prepare_ragas_dataset(csv_path, rag_service, max_questions)
    
    # Run RAGAS evaluation
    results = run_ragas_evaluation(dataset, model=ragas_model)
    
    if results:
        print_ragas_results(results, dataset)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert RAGAS results to dict for saving
        try:
            if hasattr(results, 'to_pandas'):
                ragas_scores = results.to_pandas().mean().to_dict()
            else:
                ragas_scores = {}
        except:
            ragas_scores = {}
        
        output = {
            "timestamp": timestamp,
            "mode": "advanced" if use_advanced else "simple",
            "hf_model": selected_hf_model,
            "ragas_judge_model": ragas_model,
            "total_questions": len(dataset),
            "ragas_metrics": {k: float(v) for k, v in ragas_scores.items() if v is not None},
            "dataset": dataset
        }
        
        filename = output_dir / f"ragas_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
