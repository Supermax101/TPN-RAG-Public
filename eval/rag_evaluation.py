"""
RAG System Evaluation with Industry-Standard Metrics
Supports Simple RAG and Advanced RAG (BM25 + Cross-Encoder) evaluation.

Metrics based on RAGAS framework and academic literature:
- Retrieval: Precision@K, MRR, Score Distribution
- Generation: Faithfulness, Format Correctness
- End-to-End: Accuracy, Error Classification
"""

import asyncio
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import re
from pydantic import BaseModel, Field
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.rag_metrics import (
    RAGMetricsCalculator, 
    aggregate_metrics, 
    print_recommendations,
    EvaluationSummary
)

from app.providers.embeddings import OllamaEmbeddingProvider
from app.providers.vectorstore import ChromaVectorStore
from app.providers.ollama import OllamaLLMProvider
from app.providers.openai import OpenAILLMProvider
from app.providers.xai import XAILLMProvider
from app.providers.gemini import GeminiLLMProvider
from app.providers.kimi import KimiLLMProvider
from app.services.rag import RAGService
from app.services.hybrid_rag import HybridRAGService
from app.services.advanced_rag import AdvancedRAGConfig
from app.models import SearchQuery

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class MCQAnswer(BaseModel):
    """Structured output for MCQ answers."""
    answer: str = Field(description="Single letter answer (A, B, C, D, E, or F)")
    confidence: Optional[str] = Field(default="medium", description="Confidence level")


class RAGEvaluator:
    """Evaluates RAG system with detailed metrics."""
    
    def __init__(
        self, 
        csv_path: str, 
        selected_model: str = "mistral:7b", 
        provider: str = "ollama",
        use_advanced: bool = True
    ):
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.provider = provider
        self.use_advanced = use_advanced
        self.rag_service = None
        self.evaluation_results = []
        
        self.parser = JsonOutputParser(pydantic_object=MCQAnswer)
        self.metrics_calculator = RAGMetricsCalculator(relevance_threshold=0.3)
        
        self.few_shot_examples = [
            {
                "context": "ASPEN guidelines recommend initiating PN within 24-48 hours...",
                "question": "When should PN be initiated in preterm infants?",
                "options": "A) Within 6 hours\nB) Within 24-48 hours\nC) After 72 hours\nD) After 1 week",
                "answer": '{"answer": "B", "confidence": "high"}'
            }
        ]
    
    async def initialize_rag(self):
        """Initialize RAG service based on mode."""
        embedding_provider = OllamaEmbeddingProvider()
        vector_store = ChromaVectorStore()
        
        # Select LLM provider
        if self.provider == "ollama":
            llm_provider = OllamaLLMProvider(default_model=self.selected_model)
        elif self.provider == "openai":
            llm_provider = OpenAILLMProvider(default_model=self.selected_model)
        elif self.provider == "xai":
            llm_provider = XAILLMProvider(default_model=self.selected_model)
        elif self.provider == "gemini":
            llm_provider = GeminiLLMProvider(default_model=self.selected_model)
        elif self.provider == "kimi":
            llm_provider = KimiLLMProvider(default_model=self.selected_model)
        else:
            llm_provider = OllamaLLMProvider(default_model=self.selected_model)
        
        if self.use_advanced:
            config = AdvancedRAGConfig(
                enable_bm25_hybrid=True,
                enable_cross_encoder=True,
                enable_multi_query=True,
                enable_hyde=True
            )
            self.rag_service = HybridRAGService(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                llm_provider=llm_provider,
                advanced_config=config
            )
            print(f"Mode: Advanced RAG (BM25 + Cross-Encoder Reranking)")
        else:
            self.rag_service = RAGService(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                llm_provider=llm_provider
            )
            print(f"Mode: Simple RAG (Vector Similarity Only)")
        
        # Verify collection has documents
        stats = await self.rag_service.get_collection_stats()
        if stats.get("total_chunks", 0) == 0:
            raise RuntimeError("No documents in collection. Run 'init' first.")
        
        print(f"Collection: {stats.get('total_chunks', 0)} chunks from {stats.get('total_documents', 0)} documents\n")
    
    def load_questions(self) -> pd.DataFrame:
        """Load and validate MCQ questions from CSV."""
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        
        # Expected columns: question_id, case_context, question, answer_type, options, correct_answer
        required_cols = ['question_id', 'question', 'options', 'correct_answer']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Loaded {len(df)} questions")
        multi_count = len(df[df['answer_type'] == 'multi'])
        single_count = len(df[df['answer_type'] == 'single'])
        print(f"  Single-answer: {single_count}, Multi-answer: {multi_count}")
        
        return df
    
    def build_prompt_template(self) -> ChatPromptTemplate:
        """Build simple evaluation prompt - just ask for letter answer."""
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a TPN (Total Parenteral Nutrition) Clinical Specialist taking a board-style certification examination.

KNOWLEDGE SOURCES:
1. PRIMARY: Use the provided TPN reference context (ASPEN guidelines, protocols)
2. SUPPLEMENTARY: Use your clinical training knowledge where the reference doesn't cover specifics

EXAMINATION INSTRUCTIONS:
- Options can be A, B, C, D, E, or F
- For single-answer questions: Select the ONE best answer
- For multi-answer questions: Select ALL correct answers (comma-separated)
- For "FALSE" or "LEAST likely" questions: Identify the incorrect statement

RESPONSE FORMAT: 
1. THINK: Briefly analyze the clinical context and guidelines (this is crucial for accuracy).
2. ANSWER: State the final letter choice(s).

Output format:
Thinking: [Your analysis]
Answer: [Option Letter]

--- EXAMPLES ---

Q: What is the maximum recommended dextrose concentration for peripheral IV?
A. 5% | B. 10% | C. 12.5% | D. 20%
Your answer: C

Q: Which statement about calcium in PN is FALSE?
A. Calcium gluconate is preferred | B. Monitor for precipitation | C. No limit on calcium dose | D. Check ionized calcium
Your answer: C

Q: Which components require central line access? (Select all)
A. Dextrose >12.5% | B. High osmolarity solutions | C. 10% dextrose | D. Amino acids >4%
Your answer: A,B

Your answer: Thinking: The guideline states max dextrose for peripheral is 12.5%.
Answer: C

--- END EXAMPLES ---

Reply with Thinking and Answer."""),
            ("human", """{case_context}

BOARD EXAMINATION QUESTION:
{question}

QUESTION TYPE: {answer_type}

OPTIONS:
{options}

---

TPN CLINICAL REFERENCE (from ASPEN guidelines and protocols):
{context}

Use the above reference as PRIMARY source. Supplement with your clinical knowledge if needed.

---

Your answer:""")
        ])
        
        return final_prompt
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer to handle edge cases."""
        answer = answer.strip().upper()
        
        if "ALL OF THE ABOVE" in answer:
            return "ALL"
        if answer in ["NONE", "NONE OF THE ABOVE"]:
            return "NONE"
        
        letters = re.findall(r'\b([A-F])\b', answer)
        if letters:
            return ",".join(sorted(letters))
        
        return answer
    
    def answers_match(self, model_answer: str, correct_answer: str, options_text: str) -> bool:
        """Check if model answer matches correct answer."""
        model_norm = self.normalize_answer(model_answer)
        correct_norm = self.normalize_answer(correct_answer)
        
        return model_norm == correct_norm
    
    async def evaluate_single_question(
        self,
        question_id: str,
        question: str,
        options: str,
        correct_option: str,
        case_context: str = "",
        answer_type: str = "single"
    ) -> Dict[str, Any]:
        """Evaluate a single MCQ question with industry-standard metrics."""
        
        # Build search query
        search_text = f"{case_context} {question}".strip() if case_context else question
        search_query = SearchQuery(query=search_text, limit=5)
        
        # Time the retrieval
        retrieval_start = time.time()
        search_response = await self.rag_service.search(search_query)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        # Get results from response
        search_results = search_response.results if hasattr(search_response, 'results') else search_response
        
        # Extract scores and calculate retrieval metrics
        scores = [r.score for r in search_results if hasattr(r, 'score')]
        retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(scores, retrieval_time_ms)
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            if hasattr(result, 'chunk'):
                source = result.chunk.metadata.get('source', 'Unknown')
                context_parts.append(f"[{i}] {result.chunk.content[:500]}...")
            elif isinstance(result, dict):
                context_parts.append(f"[{i}] {result.get('content', '')[:500]}...")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Build and invoke prompt
        prompt = self.build_prompt_template()
        
        # Format answer type for prompt
        type_instruction = "MULTI-ANSWER (Select all that apply)" if answer_type == "multi" else "SINGLE-ANSWER"
        
        formatted = prompt.format(
            case_context=case_context or "",
            question=question,
            answer_type=type_instruction,
            options=options,
            context=context
        )
        
        # Time the generation
        generation_start = time.time()
        raw_response = await self.rag_service.llm_provider.generate(formatted)
        generation_time_ms = (time.time() - generation_start) * 1000
        
        # Parse response - extract letter answer
        model_answer = None
        confidence = "medium"
        
        # Clean response (remove thinking tags if present)
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = clean_response.strip()
        
        # Try 1: Look for explicit "Answer: X" pattern (Best for CoT)
        match = re.search(r'(?:answer|choice)(?:\s+is)?[:\s]+([A-F](?:\s*,\s*[A-F])*)', clean_response, re.IGNORECASE)
        if match:
             model_answer = match.group(1).replace(" ", "").upper()

        # Try 2: Response starts with letter(s) like "A" or "A,B,C"
        if not model_answer:
            match = re.match(r'^([A-F](?:\s*,\s*[A-F])*)', clean_response.upper())
            if match:
                model_answer = match.group(1).replace(" ", "")
        
        # Try 3: Letter at start of line
        if not model_answer:
            match = re.search(r'(?:^|\n)\s*([A-F])\s*[.:\)]', clean_response, re.IGNORECASE)
            if match:
                model_answer = match.group(1).upper()
        
        # Try 4: First letter A-F found
        if not model_answer:
            match = re.search(r'\b([A-F])\b', clean_response.upper())
            if match:
                model_answer = match.group(1)
        
        if not model_answer:
            model_answer = "PARSE_ERROR"
        
        # Calculate generation metrics
        generation_metrics = self.metrics_calculator.calculate_generation_metrics(
            answer=model_answer,
            context=context,
            raw_response=raw_response,
            generation_time_ms=generation_time_ms
        )
        
        # Calculate end-to-end metrics
        e2e_metrics = self.metrics_calculator.calculate_e2e_metrics(
            predicted=model_answer,
            expected=correct_option,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics
        )
        
        return {
            "question_id": question_id,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "expected": correct_option,
            "predicted": model_answer,
            "confidence": confidence,
            "correct": e2e_metrics.is_correct,
            "partial_correct": e2e_metrics.partial_correct,
            "error_type": e2e_metrics.error_type,
            "retrieval_grade": retrieval_metrics.quality_grade(),
            "retrieval_metrics": {
                "num_docs": retrieval_metrics.num_retrieved,
                "avg_score": round(retrieval_metrics.avg_score, 3),
                "max_score": round(retrieval_metrics.max_score, 3),
                "mrr": round(retrieval_metrics.mrr, 3),
                "precision_at_k": round(retrieval_metrics.precision_at_k, 3),
                "high_quality": retrieval_metrics.high_quality_docs,
                "low_quality": retrieval_metrics.low_quality_docs,
                "retrieval_ms": round(retrieval_metrics.retrieval_time_ms, 1)
            },
            "generation_metrics": {
                "uses_context": generation_metrics.uses_context_phrases,
                "potential_hallucination": generation_metrics.potential_hallucination,
                "format_correct": generation_metrics.answer_format_correct,
                "generation_ms": round(generation_metrics.generation_time_ms, 1)
            },
            "total_time_ms": round(retrieval_metrics.retrieval_time_ms + generation_time_ms, 1)
        }
    
    async def run_evaluation(self, max_questions: int = None):
        """Run full evaluation with industry-standard metrics."""
        
        await self.initialize_rag()
        
        df = self.load_questions()
        
        if max_questions:
            df = df.head(max_questions)
        
        print(f"Evaluating {len(df)} questions...")
        print("=" * 70)
        
        correct_count = 0
        
        for idx, row in df.iterrows():
            question_id = str(row.get('question_id', idx + 1))
            question = str(row['question'])
            options = str(row['options'])
            correct = str(row['correct_answer'])
            answer_type = str(row.get('answer_type', 'single'))
            case_context = str(row.get('case_context', '')) if row.get('case_context') else ""
            
            print(f"\nQ{question_id}: {question[:70]}...")
            
            result = await self.evaluate_single_question(
                question_id, question, options, correct, case_context, answer_type
            )
            
            self.evaluation_results.append(result)
            
            if result["correct"]:
                correct_count += 1
                status = "CORRECT"
            elif result["partial_correct"]:
                status = "PARTIAL"
            else:
                status = "WRONG"
            
            # Print result with metrics
            rm = result["retrieval_metrics"]
            print(f"  {status}: {result['predicted']} (expected: {result['expected']})")
            print(f"  Retrieval [{result['retrieval_grade']}]: avg={rm['avg_score']:.3f}, MRR={rm['mrr']:.2f}, P@K={rm['precision_at_k']:.2f}")
            
            if result["error_type"]:
                print(f"  Error: {result['error_type']}")
            
            accuracy = (correct_count / (idx + 1)) * 100
            print(f"  Progress: {idx + 1}/{len(df)} | Accuracy: {accuracy:.1f}%")
        
        # Generate comprehensive summary using industry metrics
        summary = aggregate_metrics(self.evaluation_results)
        summary.print_summary()
        
        # Print improvement recommendations
        print_recommendations(summary)
        
        self._save_results(summary)
    
    def _save_results(self, summary: EvaluationSummary):
        """Save evaluation results to JSON with full metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "advanced" if self.use_advanced else "simple"
        model_safe = self.selected_model.replace(':', '_').replace('/', '_')
        
        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "metadata": {
                "timestamp": timestamp,
                "mode": mode,
                "model": self.selected_model,
                "provider": self.provider
            },
            "summary": {
                "total_questions": summary.total_questions,
                "correct": summary.correct,
                "partial_correct": summary.partial_correct,
                "accuracy": round(summary.accuracy, 2),
                "accuracy_with_partial": round(summary.accuracy_with_partial, 2),
                "avg_retrieval_score": round(summary.avg_retrieval_score, 3),
                "retrieval_grades": summary.retrieval_grade_distribution,
                "error_types": summary.error_distribution,
                "avg_retrieval_ms": round(summary.avg_retrieval_time_ms, 1),
                "avg_generation_ms": round(summary.avg_generation_time_ms, 1)
            },
            "detailed_results": self.evaluation_results
        }
        
        filename = output_dir / f"eval_{mode}_{model_safe}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved: {filename}")


async def get_available_ollama_models():
    """Get list of available Ollama LLM models."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                all_models = [model["name"] for model in data.get("models", [])]
                
                embedding_keywords = ["embed", "embedding", "nomic-embed"]
                llm_models = [
                    model for model in all_models 
                    if not any(keyword in model.lower() for keyword in embedding_keywords)
                ]
                
                return llm_models
    except Exception:
        pass
    return []


async def get_all_available_models():
    """Get all available models from all providers."""
    from app.config import settings
    
    all_models = []
    
    ollama_models = await get_available_ollama_models()
    if ollama_models:
        all_models.extend([("ollama", model) for model in ollama_models])
    
    if settings.openai_api_key:
        all_models.extend([("openai", "gpt-4o")])
    
    if settings.gemini_api_key:
        all_models.extend([("gemini", "gemini-2.5-flash"), ("gemini", "gemini-2.5-pro")])
    
    if settings.xai_api_key:
        all_models.extend([("xai", "grok-4-fast-reasoning")])
    
    if settings.kimi_api_key:
        all_models.extend([("kimi", "kimi-k2-0905-preview")])
    
    return all_models


def select_model(available_models):
    """Interactive model selection."""
    if not available_models:
        print("\nERROR: No LLM models found.")
        return None, None
    
    print(f"\nAvailable LLM Models ({len(available_models)} found):")
    
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"  {i}. [{provider.upper()}] {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or Enter for default: ").strip()
            
            if not choice:
                return available_models[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                return available_models[choice_num - 1]
        except ValueError:
            print("Please enter a valid number")


async def main():
    """Main evaluation function."""
    
    print("\n" + "=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    # Select RAG mode
    print("\nEvaluation Modes:")
    print("  1. Simple RAG (vector similarity only)")
    print("  2. Advanced RAG (BM25 + Cross-Encoder reranking)")
    
    mode_choice = input("\nSelect mode (1 or 2, default=1): ").strip()
    use_advanced = mode_choice == "2"
    
    # Select model
    available_models = await get_all_available_models()
    if not available_models:
        print("No models available")
        return
    
    selected_provider, selected_model = select_model(available_models)
    if not selected_provider:
        return
    
    print(f"\nSelected: [{selected_provider.upper()}] {selected_model}")
    
    # Question limit
    max_questions_input = input(f"\nLimit questions? (Enter for all, or number): ").strip()
    max_questions = int(max_questions_input) if max_questions_input.isdigit() else None
    
    try:
        evaluator = RAGEvaluator(
            csv_path, 
            selected_model, 
            selected_provider,
            use_advanced=use_advanced
        )
        await evaluator.run_evaluation(max_questions)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
