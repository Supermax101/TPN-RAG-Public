"""
Production RAG Evaluation v2 - LangChain 1.x

This is the refactored evaluation script using:
- New semantic chunking pipeline
- LangChain 1.x MCQ Chain with structured output
- Improved metrics and error analysis

Usage:
    uv run python eval/rag_evaluation_v2.py
"""

import asyncio
import pandas as pd
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich import box

from app.chains.mcq_chain import MCQChain, MCQChainConfig
from app.chains.retrieval_chain import RetrievalChain, RetrievalConfig
from app.parsers.mcq_parser import normalize_answer, answers_match
from app.config import settings
from app.logger import logger

console = Console()


@dataclass
class EvaluationResult:
    """Result for a single question evaluation."""
    question_id: str
    question: str
    expected: str
    predicted: str
    correct: bool
    partial_correct: bool
    confidence: str
    thinking: str
    error_type: Optional[str]
    retrieval_scores: List[float]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


@dataclass
class EvaluationSummary:
    """Aggregated evaluation summary."""
    total_questions: int = 0
    correct: int = 0
    partial_correct: int = 0
    accuracy: float = 0.0
    accuracy_with_partial: float = 0.0
    avg_retrieval_score: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    def print_summary(self):
        """Print formatted evaluation summary."""
        print("\n" + "=" * 75)
        print("                    RAG EVALUATION SUMMARY (v2)")
        print("                    LangChain 1.x + Semantic Chunking")
        print("=" * 75)
        
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        table.add_column("Metric", width=30)
        table.add_column("Value", justify="right", width=15)
        table.add_column("Details", width=25)
        
        table.add_row(
            "Accuracy (Exact)",
            f"{self.accuracy:.1f}%",
            f"{self.correct}/{self.total_questions} correct"
        )
        table.add_row(
            "Accuracy (with Partial)",
            f"{self.accuracy_with_partial:.1f}%",
            f"+{self.partial_correct} partial"
        )
        table.add_row(
            "Avg Retrieval Score",
            f"{self.avg_retrieval_score:.3f}",
            self._score_grade(self.avg_retrieval_score)
        )
        table.add_row("Avg Retrieval Time", f"{self.avg_retrieval_time_ms:.0f} ms", "")
        table.add_row("Avg Generation Time", f"{self.avg_generation_time_ms:.0f} ms", "")
        table.add_row("Avg Total Time", f"{self.avg_total_time_ms:.0f} ms", "")
        
        console.print(table)
        
        # Error distribution
        if self.error_distribution:
            print("\n" + "-" * 75)
            print("Error Distribution")
            print("-" * 75)
            
            error_table = Table(box=box.SIMPLE, show_header=True)
            error_table.add_column("Error Type", width=25)
            error_table.add_column("Count", justify="right", width=10)
            error_table.add_column("Percentage", justify="right", width=12)
            
            total_errors = sum(self.error_distribution.values())
            for error_type, count in sorted(self.error_distribution.items(), key=lambda x: -x[1]):
                pct = (count / total_errors * 100) if total_errors > 0 else 0
                error_table.add_row(error_type, str(count), f"{pct:.0f}%")
            
            console.print(error_table)
        
        print("=" * 75)
    
    def _score_grade(self, score: float) -> str:
        if score >= 0.7: return "Excellent"
        if score >= 0.5: return "Good"
        if score >= 0.3: return "Moderate"
        return "Poor"


class RAGEvaluatorV2:
    """
    Production RAG Evaluator using LangChain 1.x pipeline.
    """
    
    def __init__(
        self,
        csv_path: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        retrieval_k: int = 5,
        enable_reranking: bool = True,
    ):
        self.csv_path = csv_path
        self.model = model
        self.retrieval_k = retrieval_k
        self.enable_reranking = enable_reranking
        
        self.mcq_chain: Optional[MCQChain] = None
        self.results: List[EvaluationResult] = []
    
    async def initialize(self):
        """Initialize the evaluation pipeline."""
        console.print("[cyan]Initializing LangChain 1.x pipeline...[/cyan]")
        
        # Create MCQ chain
        config = MCQChainConfig(
            model=self.model,
            retrieval_k=self.retrieval_k,
            enable_reranking=self.enable_reranking,
            use_structured_output=True,
        )
        
        self.mcq_chain = MCQChain(config=config)
        await self.mcq_chain.initialize()
        
        # Print stats
        stats = self.mcq_chain.get_stats()
        retrieval_stats = stats.get("retrieval", {})
        
        console.print(f"[green]✓ Model: {self.model}[/green]")
        console.print(f"[green]✓ Documents: {retrieval_stats.get('total_documents', 0)} chunks[/green]")
        console.print(f"[green]✓ BM25: {'enabled' if retrieval_stats.get('bm25_enabled') else 'disabled'}[/green]")
        console.print(f"[green]✓ Reranker: {'enabled' if retrieval_stats.get('reranker_enabled') else 'disabled'}[/green]")
    
    def load_questions(self) -> pd.DataFrame:
        """Load MCQ questions from CSV."""
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        
        required = ['question_id', 'question', 'options', 'correct_answer']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        console.print(f"\n[cyan]Loaded {len(df)} questions[/cyan]")
        
        multi_count = len(df[df.get('answer_type', 'single') == 'multi'])
        single_count = len(df) - multi_count
        console.print(f"  Single-answer: {single_count}, Multi-answer: {multi_count}")
        
        return df
    
    async def evaluate_question(
        self,
        question_id: str,
        question: str,
        options: str,
        correct_answer: str,
        case_context: str = "",
        answer_type: str = "single",
    ) -> EvaluationResult:
        """Evaluate a single MCQ question."""
        
        start_time = time.time()
        retrieval_start = time.time()
        
        # Get answer from MCQ chain
        result = await self.mcq_chain.answer(
            question=question,
            options=options,
            answer_type=answer_type,
            case_context=case_context,
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Normalize and compare
        predicted = result.get("answer", "ERROR")
        expected = normalize_answer(correct_answer)
        predicted_norm = normalize_answer(predicted)
        
        exact_match, partial_match = answers_match(predicted, correct_answer)
        
        # Classify error
        error_type = None
        if not exact_match:
            scores = result.get("retrieval_scores", [])
            avg_score = sum(scores) / len(scores) if scores else 0
            
            if avg_score < 0.3:
                error_type = "retrieval_failure"
            elif predicted in ["ERROR", "PARSE_ERROR"]:
                error_type = "parse_error"
            elif partial_match:
                error_type = "multi_answer_partial"
            else:
                error_type = "generation_failure"
        
        return EvaluationResult(
            question_id=question_id,
            question=question[:100] + "..." if len(question) > 100 else question,
            expected=expected,
            predicted=predicted_norm,
            correct=exact_match,
            partial_correct=partial_match,
            confidence=result.get("confidence", "unknown"),
            thinking=result.get("thinking", "")[:200],
            error_type=error_type,
            retrieval_scores=result.get("retrieval_scores", []),
            retrieval_time_ms=0,  # Not tracked separately in new chain
            generation_time_ms=0,
            total_time_ms=total_time_ms,
        )
    
    async def run_evaluation(self, max_questions: Optional[int] = None) -> EvaluationSummary:
        """Run full evaluation."""
        
        await self.initialize()
        df = self.load_questions()
        
        if max_questions:
            df = df.head(max_questions)
        
        console.print(f"\n[bold]Evaluating {len(df)} questions...[/bold]")
        console.print("=" * 70)
        
        for idx, row in df.iterrows():
            question_id = str(row.get('question_id', idx + 1))
            question = str(row['question'])
            options = str(row['options'])
            correct = str(row['correct_answer'])
            answer_type = str(row.get('answer_type', 'single'))
            case_context = str(row.get('case_context', '')) if row.get('case_context') else ""
            
            console.print(f"\nQ{question_id}: {question[:60]}...")
            
            result = await self.evaluate_question(
                question_id=question_id,
                question=question,
                options=options,
                correct_answer=correct,
                case_context=case_context,
                answer_type=answer_type,
            )
            
            self.results.append(result)
            
            # Print result
            if result.correct:
                status = "[green]CORRECT[/green]"
            elif result.partial_correct:
                status = "[yellow]PARTIAL[/yellow]"
            else:
                status = "[red]WRONG[/red]"
            
            console.print(f"  {status}: {result.predicted} (expected: {result.expected})")
            
            if result.error_type:
                console.print(f"  [dim]Error: {result.error_type}[/dim]")
            
            # Progress
            current = idx + 1
            correct_so_far = sum(1 for r in self.results if r.correct)
            accuracy = (correct_so_far / current) * 100
            console.print(f"  [dim]Progress: {current}/{len(df)} | Accuracy: {accuracy:.1f}%[/dim]")
        
        # Calculate summary
        summary = self._calculate_summary()
        summary.print_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _calculate_summary(self) -> EvaluationSummary:
        """Calculate evaluation summary."""
        summary = EvaluationSummary()
        summary.total_questions = len(self.results)
        
        if not self.results:
            return summary
        
        summary.correct = sum(1 for r in self.results if r.correct)
        summary.partial_correct = sum(1 for r in self.results if r.partial_correct and not r.correct)
        
        summary.accuracy = (summary.correct / summary.total_questions) * 100
        summary.accuracy_with_partial = ((summary.correct + summary.partial_correct) / summary.total_questions) * 100
        
        # Retrieval scores
        all_scores = []
        for r in self.results:
            all_scores.extend(r.retrieval_scores)
        if all_scores:
            summary.avg_retrieval_score = sum(all_scores) / len(all_scores)
        
        # Timing
        summary.avg_total_time_ms = sum(r.total_time_ms for r in self.results) / len(self.results)
        
        # Error distribution
        for r in self.results:
            if r.error_type:
                summary.error_distribution[r.error_type] = summary.error_distribution.get(r.error_type, 0) + 1
        
        return summary
    
    def _save_results(self, summary: EvaluationSummary):
        """Save evaluation results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = self.model.replace(':', '_').replace('/', '_')
        
        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "metadata": {
                "timestamp": timestamp,
                "model": self.model,
                "retrieval_k": self.retrieval_k,
                "reranking_enabled": self.enable_reranking,
                "pipeline": "LangChain 1.x + SemanticChunking",
            },
            "summary": {
                "total_questions": summary.total_questions,
                "correct": summary.correct,
                "partial_correct": summary.partial_correct,
                "accuracy": round(summary.accuracy, 2),
                "accuracy_with_partial": round(summary.accuracy_with_partial, 2),
                "avg_retrieval_score": round(summary.avg_retrieval_score, 3),
                "error_distribution": summary.error_distribution,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "partial_correct": r.partial_correct,
                    "error_type": r.error_type,
                    "confidence": r.confidence,
                    "total_time_ms": round(r.total_time_ms, 1),
                }
                for r in self.results
            ]
        }
        
        filename = output_dir / f"eval_v2_{model_safe}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        console.print(f"\n[green]Results saved: {filename}[/green]")


async def select_model() -> tuple[str, str]:
    """Interactive model selection."""
    all_models = []

    # Add HuggingFace models (primary provider)
    if settings.hf_token:
        all_models.extend([
            ("huggingface", "Qwen/Qwen2.5-7B-Instruct"),
            ("huggingface", "Qwen/Qwen2.5-14B-Instruct"),
            ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
            ("huggingface", "mistralai/Mistral-7B-Instruct-v0.3"),
        ])
        # Add configured model if different
        if settings.hf_llm_model and ("huggingface", settings.hf_llm_model) not in all_models:
            all_models.insert(0, ("huggingface", settings.hf_llm_model))

    # Add cloud providers if configured
    if settings.openai_api_key:
        all_models.append(("openai", "gpt-4o"))
    if settings.gemini_api_key:
        all_models.append(("gemini", "gemini-2.0-flash"))

    if not all_models:
        console.print("[red]No models available! Set HF_TOKEN or other API keys.[/red]")
        return None, None

    console.print(f"\n[bold]Available Models ({len(all_models)}):[/bold]")
    for i, (provider, model) in enumerate(all_models, 1):
        console.print(f"  {i}. [{provider.upper()}] {model}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(all_models)}, default=1): ").strip()
            if not choice:
                return all_models[0]

            choice_num = int(choice)
            if 1 <= choice_num <= len(all_models):
                return all_models[choice_num - 1]
        except ValueError:
            pass


async def main():
    """Main evaluation function."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]RAG EVALUATION v2 - LangChain 1.x Pipeline[/bold cyan]")
    console.print("=" * 70)
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    # Model selection
    provider, model = await select_model()
    if not model:
        return
    
    console.print(f"\n[green]Selected: [{provider.upper()}] {model}[/green]")
    
    # Question limit
    limit_input = input("\nLimit questions? (Enter for all, or number): ").strip()
    max_questions = int(limit_input) if limit_input.isdigit() else None
    
    try:
        evaluator = RAGEvaluatorV2(
            csv_path=csv_path,
            model=model,
            retrieval_k=5,
            enable_reranking=True,
        )
        
        await evaluator.run_evaluation(max_questions)
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
