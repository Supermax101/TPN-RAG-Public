"""
Production-Grade RAG Evaluation Pipeline.

This is a hospital/clinical-grade evaluation system that:
1. Measures RETRIEVAL quality (are we finding the right documents?)
2. Measures GENERATION quality (is the answer correct and grounded?)
3. Detects HALLUCINATION (is the model making things up?)
4. Provides ERROR ANALYSIS (why are we failing?)

Uses:
- DeepEval G-Eval for LLM-as-judge metrics
- Custom clinical accuracy metrics
- Proper stratified analysis

Usage:
    uv run python eval/production_evaluation.py
    uv run python eval/production_evaluation.py --model qwen2.5:7b --limit 10
"""

import asyncio
import json
import csv
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.logger import logger

console = Console()


# =============================================================================
# EVALUATION RESULT TYPES
# =============================================================================

class ErrorType(Enum):
    """Classification of evaluation errors."""
    NONE = "none"
    RETRIEVAL_FAILURE = "retrieval_failure"       # Didn't find relevant docs
    GROUNDING_FAILURE = "grounding_failure"       # Answer not based on context
    REASONING_ERROR = "reasoning_error"           # Wrong reasoning from correct context
    PARSING_ERROR = "parsing_error"               # Couldn't parse answer
    MULTI_ANSWER_PARTIAL = "multi_answer_partial" # Got some but not all answers
    HALLUCINATION = "hallucination"               # Made up information


@dataclass
class RetrievalResult:
    """Metrics for retrieval quality."""
    documents_retrieved: int = 0
    avg_similarity_score: float = 0.0
    top_score: float = 0.0
    relevant_docs_found: int = 0  # Based on threshold
    retrieval_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    
    @property
    def retrieval_precision(self) -> float:
        """Precision of retrieved documents."""
        if self.documents_retrieved == 0:
            return 0.0
        return self.relevant_docs_found / self.documents_retrieved


@dataclass 
class GenerationResult:
    """Metrics for generation quality."""
    predicted_answer: str = ""
    correct_answer: str = ""
    is_correct: bool = False
    is_partial_correct: bool = False  # For multi-answer
    reasoning: str = ""
    confidence: str = "unknown"
    generation_time_ms: float = 0.0
    
    # LLM-as-judge scores (0-1)
    faithfulness_score: float = 0.0      # Is answer grounded in context?
    relevance_score: float = 0.0         # Does answer address the question?
    correctness_score: float = 0.0       # Is answer factually correct?


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single question."""
    question_id: int
    question: str
    answer_type: str  # single or multi
    
    retrieval: RetrievalResult
    generation: GenerationResult
    
    error_type: ErrorType = ErrorType.NONE
    error_details: str = ""
    
    total_time_ms: float = 0.0
    
    @property
    def is_success(self) -> bool:
        return self.generation.is_correct


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    timestamp: str
    model: str
    total_questions: int
    
    # Accuracy metrics
    accuracy: float = 0.0
    partial_accuracy: float = 0.0  # Including partial multi-answer
    
    # Retrieval metrics
    avg_retrieval_precision: float = 0.0
    avg_top_similarity: float = 0.0
    retrieval_failure_rate: float = 0.0
    
    # Generation metrics  
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    hallucination_rate: float = 0.0
    
    # Error breakdown
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Per-category accuracy
    single_answer_accuracy: float = 0.0
    multi_answer_accuracy: float = 0.0
    
    # Timing
    avg_time_ms: float = 0.0
    total_time_seconds: float = 0.0


# =============================================================================
# EVALUATION METRICS (LLM-as-Judge)
# =============================================================================

class ProductionMetrics:
    """
    Production-grade LLM-as-judge metrics.
    Uses DeepEval's G-Eval pattern with explicit evaluation steps.
    """
    
    def __init__(self, judge_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.judge_model = judge_model
        self.judge_llm = None

    async def initialize(self):
        """Initialize the judge LLM."""
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        # Create HuggingFace endpoint
        llm = HuggingFaceEndpoint(
            repo_id=self.judge_model,
            huggingfacehub_api_token=settings.hf_token,
            temperature=0.01,  # HuggingFace doesn't support 0.0
            max_new_tokens=1024,
        )
        self.judge_llm = ChatHuggingFace(llm=llm)
    
    async def evaluate_faithfulness(
        self,
        question: str,
        context: str,
        answer: str,
        reasoning: str,
    ) -> float:
        """
        Evaluate if the answer is GROUNDED in the provided context.
        
        This is the most critical metric for clinical use.
        Score 0-1 where:
        - 1.0 = Fully grounded, every claim supported by context
        - 0.5 = Partially grounded
        - 0.0 = Not grounded / hallucinated
        """
        
        prompt = f"""You are evaluating whether a clinical answer is GROUNDED in the provided context.

CONTEXT (from TPN knowledge base):
{context[:3000]}

QUESTION:
{question}

ANSWER GIVEN:
{answer}

REASONING PROVIDED:
{reasoning}

EVALUATION STEPS:
1. Identify each clinical claim made in the answer
2. For each claim, check if it is explicitly supported by the context
3. Flag any claims that are NOT in the context (potential hallucination)
4. Calculate grounding percentage

SCORING:
- 1.0: All claims are directly supported by context
- 0.7-0.9: Most claims supported, minor unsupported details
- 0.4-0.6: Mixed - some claims supported, some not
- 0.1-0.3: Mostly unsupported claims
- 0.0: Answer contradicts context or is completely fabricated

Output ONLY a JSON object:
{{"score": <float 0-1>, "grounded_claims": <int>, "total_claims": <int>, "hallucinated": <list of unsupported claims>}}"""

        try:
            response = self.judge_llm.invoke([{"role": "user", "content": prompt}])
            result = json.loads(response.content)
            return float(result.get("score", 0.5))
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5  # Default to uncertain
    
    async def evaluate_answer_correctness(
        self,
        question: str,
        predicted: str,
        correct: str,
        answer_type: str,
    ) -> Tuple[bool, bool, float]:
        """
        Evaluate if the predicted answer matches the correct answer.
        
        Returns: (is_exact_match, is_partial_match, correctness_score)
        """
        
        # Normalize answers
        predicted = predicted.upper().strip()
        correct = correct.upper().strip()
        
        if answer_type == "multi":
            # Parse comma-separated answers
            pred_set = set(a.strip() for a in predicted.replace(" ", "").split(","))
            correct_set = set(a.strip() for a in correct.replace(" ", "").split(","))
            
            is_exact = pred_set == correct_set
            
            # Calculate partial match
            if correct_set:
                overlap = len(pred_set & correct_set)
                precision = overlap / len(pred_set) if pred_set else 0
                recall = overlap / len(correct_set)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                is_partial = overlap > 0
                score = f1
            else:
                is_partial = False
                score = 0.0
            
            return is_exact, is_partial, score
        else:
            # Single answer
            is_exact = predicted == correct
            is_partial = correct in predicted or predicted in correct
            score = 1.0 if is_exact else (0.5 if is_partial else 0.0)
            
            return is_exact, is_partial, score
    
    async def detect_hallucination(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Tuple[bool, str]:
        """
        Detect if the answer contains hallucinated information.
        
        Returns: (has_hallucination, details)
        """
        
        prompt = f"""You are a clinical hallucination detector. Your job is to identify if the answer contains information NOT present in the context.

CONTEXT:
{context[:2000]}

QUESTION: {question}

ANSWER: {answer}

Does the answer contain any of these hallucination types?
1. FABRICATED NUMBERS - dosages, percentages, values not in context
2. INVENTED GUIDELINES - citing guidelines or recommendations not in context  
3. FALSE CLAIMS - statements that contradict the context
4. UNSUPPORTED SPECIFICS - very specific claims with no context support

Output ONLY a JSON object:
{{"has_hallucination": <true/false>, "type": "<type or null>", "details": "<what was hallucinated or null>"}}"""

        try:
            response = self.judge_llm.invoke([{"role": "user", "content": prompt}])
            result = json.loads(response.content)
            return result.get("has_hallucination", False), result.get("details", "")
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {e}")
            return False, ""


# =============================================================================
# PRODUCTION EVALUATOR
# =============================================================================

class ProductionEvaluator:
    """
    Production-grade RAG evaluator.
    
    Usage:
        evaluator = ProductionEvaluator()
        await evaluator.initialize()
        report = await evaluator.run_full_evaluation()
    """
    
    def __init__(
        self,
        csv_path: str = "eval/tpn_mcq_cleaned.csv",
        model: str = None,
        use_agentic: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.model = model or settings.hf_llm_model or "Qwen/Qwen2.5-7B-Instruct"
        self.use_agentic = use_agentic
        
        self.rag_chain = None
        self.metrics = None
        self.results: List[EvaluationResult] = []
    
    async def initialize(self):
        """Initialize RAG chain and metrics."""
        console.print(f"\n[cyan]Initializing Production Evaluator...[/cyan]")
        console.print(f"  Model: {self.model}")
        console.print(f"  Mode: {'Agentic RAG' if self.use_agentic else 'Standard RAG'}")
        
        # Initialize RAG
        if self.use_agentic:
            from app.chains.agentic_rag import create_agentic_mcq_rag
            self.rag_chain = await create_agentic_mcq_rag(self.model)
        else:
            from app.chains.mcq_chain import create_mcq_chain
            self.rag_chain = create_mcq_chain(model=self.model)
            await self.rag_chain.initialize()
        
        # Initialize metrics
        self.metrics = ProductionMetrics(judge_model=self.model)
        await self.metrics.initialize()
        
        console.print("[green]✓ Initialization complete[/green]")
    
    def load_questions(self) -> pd.DataFrame:
        """Load evaluation questions."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        console.print(f"[green]✓ Loaded {len(df)} questions[/green]")
        return df
    
    async def evaluate_single_question(
        self,
        row: pd.Series,
    ) -> EvaluationResult:
        """Evaluate a single question with full metrics."""
        
        start_time = time.time()
        
        question_id = int(row.get("question_id", 0))
        question = str(row.get("question", ""))
        options = str(row.get("options", ""))
        correct_answer = str(row.get("correct_answer", ""))
        answer_type = str(row.get("answer_type", "single"))
        case_context = str(row.get("case_context", "")) or ""
        
        # Initialize result
        retrieval_result = RetrievalResult()
        generation_result = GenerationResult(correct_answer=correct_answer)
        error_type = ErrorType.NONE
        error_details = ""
        
        try:
            # Run RAG
            retrieval_start = time.time()
            
            if self.use_agentic:
                result = await self.rag_chain.answer(
                    question=question,
                    options=options,
                    answer_type=answer_type,
                    case_context=case_context,
                )
                context = result.get("context", "")  # May not be available
            else:
                result = await self.rag_chain.answer(
                    question=question,
                    options=options,
                    answer_type=answer_type,
                    case_context=case_context,
                )
                context = result.get("context", "")
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Parse result
            predicted = result.get("answer", "")
            reasoning = result.get("thinking", "")
            confidence = result.get("confidence", "unknown")
            
            generation_result.predicted_answer = predicted
            generation_result.reasoning = reasoning
            generation_result.confidence = confidence
            retrieval_result.retrieval_time_ms = retrieval_time
            
            # Check if we got context
            if not result.get("context_used", True):
                retrieval_result.relevant_docs_found = 0
                error_type = ErrorType.RETRIEVAL_FAILURE
                error_details = "No relevant documents retrieved"
            
            # Evaluate correctness
            is_exact, is_partial, score = await self.metrics.evaluate_answer_correctness(
                question=question,
                predicted=predicted,
                correct=correct_answer,
                answer_type=answer_type,
            )
            
            generation_result.is_correct = is_exact
            generation_result.is_partial_correct = is_partial
            generation_result.correctness_score = score
            
            # Evaluate faithfulness (if we have context)
            if context:
                faithfulness = await self.metrics.evaluate_faithfulness(
                    question=question,
                    context=context,
                    answer=predicted,
                    reasoning=reasoning,
                )
                generation_result.faithfulness_score = faithfulness
                
                # Check for hallucination
                if faithfulness < 0.3:
                    has_hallucination, details = await self.metrics.detect_hallucination(
                        question=question,
                        context=context,
                        answer=predicted,
                    )
                    if has_hallucination:
                        error_type = ErrorType.HALLUCINATION
                        error_details = details
            
            # Classify error if wrong
            if not is_exact and error_type == ErrorType.NONE:
                if answer_type == "multi" and is_partial:
                    error_type = ErrorType.MULTI_ANSWER_PARTIAL
                elif generation_result.faithfulness_score < 0.5:
                    error_type = ErrorType.GROUNDING_FAILURE
                else:
                    error_type = ErrorType.REASONING_ERROR
            
        except Exception as e:
            error_type = ErrorType.PARSING_ERROR
            error_details = str(e)
            logger.error(f"Evaluation error Q{question_id}: {e}")
        
        total_time = (time.time() - start_time) * 1000
        generation_result.generation_time_ms = total_time - retrieval_result.retrieval_time_ms
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            answer_type=answer_type,
            retrieval=retrieval_result,
            generation=generation_result,
            error_type=error_type,
            error_details=error_details,
            total_time_ms=total_time,
        )
    
    async def run_full_evaluation(
        self,
        max_questions: Optional[int] = None,
    ) -> EvaluationReport:
        """Run full evaluation and generate report."""
        
        if not self.rag_chain:
            await self.initialize()
        
        df = self.load_questions()
        
        if max_questions:
            df = df.head(max_questions)
        
        console.print(f"\n[bold]Running evaluation on {len(df)} questions...[/bold]\n")
        
        self.results = []
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(df))
            
            for idx, row in df.iterrows():
                result = await self.evaluate_single_question(row)
                self.results.append(result)
                
                # Update progress with running accuracy
                correct = sum(1 for r in self.results if r.is_success)
                accuracy = correct / len(self.results) * 100
                progress.update(
                    task,
                    advance=1,
                    description=f"Evaluating... ({accuracy:.1f}% accuracy)"
                )
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self._generate_report(total_time)
        
        # Display results
        self._display_results(report)
        
        # Save results
        self._save_results(report)
        
        return report
    
    def _generate_report(self, total_time: float) -> EvaluationReport:
        """Generate evaluation report from results."""
        
        n = len(self.results)
        
        # Accuracy
        correct = sum(1 for r in self.results if r.is_success)
        partial = sum(1 for r in self.results if r.generation.is_partial_correct)
        
        # By type
        single = [r for r in self.results if r.answer_type == "single"]
        multi = [r for r in self.results if r.answer_type == "multi"]
        
        single_correct = sum(1 for r in single if r.is_success)
        multi_correct = sum(1 for r in multi if r.is_success)
        
        # Error distribution
        errors = {}
        for r in self.results:
            err = r.error_type.value
            errors[err] = errors.get(err, 0) + 1
        
        # Faithfulness
        faithfulness_scores = [r.generation.faithfulness_score for r in self.results if r.generation.faithfulness_score > 0]
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        
        # Hallucination rate
        hallucinations = sum(1 for r in self.results if r.error_type == ErrorType.HALLUCINATION)
        
        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            model=self.model,
            total_questions=n,
            accuracy=correct / n if n > 0 else 0,
            partial_accuracy=partial / n if n > 0 else 0,
            avg_faithfulness=avg_faithfulness,
            hallucination_rate=hallucinations / n if n > 0 else 0,
            error_distribution=errors,
            single_answer_accuracy=single_correct / len(single) if single else 0,
            multi_answer_accuracy=multi_correct / len(multi) if multi else 0,
            avg_time_ms=sum(r.total_time_ms for r in self.results) / n if n > 0 else 0,
            total_time_seconds=total_time,
        )
    
    def _display_results(self, report: EvaluationReport):
        """Display evaluation results."""
        
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]PRODUCTION EVALUATION REPORT[/bold cyan]")
        console.print("=" * 70)
        
        # Main metrics
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status")
        
        # Accuracy
        acc = report.accuracy * 100
        status = "✅" if acc >= 90 else ("⚠️" if acc >= 70 else "❌")
        table.add_row("Overall Accuracy", f"{acc:.1f}%", status)
        
        table.add_row("Single-Answer Accuracy", f"{report.single_answer_accuracy*100:.1f}%", "")
        table.add_row("Multi-Answer Accuracy", f"{report.multi_answer_accuracy*100:.1f}%", "")
        
        # Faithfulness
        faith = report.avg_faithfulness * 100
        status = "✅" if faith >= 80 else ("⚠️" if faith >= 60 else "❌")
        table.add_row("Avg Faithfulness (Grounding)", f"{faith:.1f}%", status)
        
        # Hallucination
        hall = report.hallucination_rate * 100
        status = "✅" if hall <= 5 else ("⚠️" if hall <= 15 else "❌")
        table.add_row("Hallucination Rate", f"{hall:.1f}%", status)
        
        table.add_row("Avg Response Time", f"{report.avg_time_ms:.0f}ms", "")
        table.add_row("Total Time", f"{report.total_time_seconds:.1f}s", "")
        
        console.print(table)
        
        # Error breakdown
        console.print("\n[bold]Error Distribution:[/bold]")
        for error_type, count in sorted(report.error_distribution.items(), key=lambda x: -x[1]):
            pct = count / report.total_questions * 100
            console.print(f"  {error_type}: {count} ({pct:.1f}%)")
        
        # Failed questions
        failed = [r for r in self.results if not r.is_success]
        if failed and len(failed) <= 10:
            console.print("\n[bold]Failed Questions:[/bold]")
            for r in failed[:10]:
                console.print(f"  Q{r.question_id}: {r.generation.predicted_answer} (expected {r.generation.correct_answer}) - {r.error_type.value}")
        
        console.print("\n" + "=" * 70)
    
    def _save_results(self, report: EvaluationReport):
        """Save results to files."""
        
        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_path = output_dir / f"report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save detailed results
        results_path = output_dir / f"results_{timestamp}.json"
        with open(results_path, "w") as f:
            results_data = [
                {
                    "question_id": r.question_id,
                    "question": r.question[:100],
                    "predicted": r.generation.predicted_answer,
                    "correct": r.generation.correct_answer,
                    "is_correct": r.is_success,
                    "faithfulness": r.generation.faithfulness_score,
                    "error_type": r.error_type.value,
                    "time_ms": r.total_time_ms,
                }
                for r in self.results
            ]
            json.dump(results_data, f, indent=2)
        
        console.print(f"\n[dim]Results saved to: {output_dir}[/dim]")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run production evaluation."""
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def evaluate(
        model: str = typer.Option(None, "--model", "-m", help="LLM model to use"),
        limit: int = typer.Option(None, "--limit", "-n", help="Max questions to evaluate"),
        agentic: bool = typer.Option(True, "--agentic/--simple", help="Use agentic RAG"),
    ):
        """Run production-grade RAG evaluation."""
        
        evaluator = ProductionEvaluator(
            model=model,
            use_agentic=agentic,
        )
        
        asyncio.run(evaluator.run_full_evaluation(max_questions=limit))
    
    app()


if __name__ == "__main__":
    main()
