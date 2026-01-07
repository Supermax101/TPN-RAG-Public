#!/usr/bin/env python3
"""
Full Evaluation Suite with JSON Output and Academic Graphs
Runs comprehensive evaluation and generates publication-ready outputs.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import pandas as pd
import numpy as np

# Try to import matplotlib for graphs
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, graphs will be skipped")


@dataclass
class EvalResult:
    """Single evaluation result."""
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    is_correct: bool
    retrieval_hit: bool
    f1_score: float
    response_time_ms: float
    num_chunks_retrieved: int
    source_match: bool


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    timestamp: str
    model_name: str
    embedding_model: str
    total_questions: int
    correct_answers: int
    accuracy: float
    retrieval_hit_rate: float
    mean_f1: float
    mean_response_time_ms: float
    total_time_seconds: float
    config: Dict[str, Any]


class FullEvaluator:
    """Comprehensive evaluation with JSON and graph outputs."""

    def __init__(
        self,
        model_name: str = "chandramax/tpn-gpt-oss-20b",
        embedding_model: str = "Qwen/Qwen3-Embedding-8B",
        output_dir: str = "eval/results",
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[EvalResult] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_test_data(self, path: str) -> List[Dict]:
        """Load test questions from JSONL."""
        questions = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions

    def load_mcq_data(self, csv_path: str) -> pd.DataFrame:
        """Load MCQ questions from CSV."""
        df = pd.read_csv(csv_path, keep_default_na=False)
        return df

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip().upper()
        # Extract letters only
        letters = re.findall(r'\b([A-F])\b', answer)
        if letters:
            return ",".join(sorted(set(letters)))
        return answer

    def calculate_f1(self, predicted: str, reference: str) -> float:
        """Calculate F1 score between predicted and reference answers."""
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    async def evaluate_single(
        self,
        question: str,
        expected: str,
        question_id: str,
        vectorstore,
        llm_provider,
    ) -> EvalResult:
        """Evaluate a single question."""
        start_time = time.time()

        try:
            # Retrieve context
            docs = vectorstore.similarity_search(question, k=10)
            num_chunks = len(docs)
            retrieval_hit = num_chunks > 0

            # Check source match (simplified)
            source_match = False

            # Build context
            context = "\n\n---\n\n".join([
                f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
                for d in docs
            ])

            # Generate answer using the LLMProvider interface
            response = llm_provider.generate(
                question=question,
                context=context,
            )

            generated = response.answer if hasattr(response, 'answer') else str(response)

            response_time = (time.time() - start_time) * 1000

            # Calculate metrics
            f1 = self.calculate_f1(generated, expected)

            # Simple correctness check
            expected_norm = self.normalize_answer(expected)
            generated_norm = self.normalize_answer(generated)
            is_correct = expected_norm == generated_norm or f1 > 0.7

            return EvalResult(
                question_id=question_id,
                question=question,
                expected_answer=expected,
                generated_answer=generated[:500],
                is_correct=is_correct,
                retrieval_hit=retrieval_hit,
                f1_score=f1,
                response_time_ms=response_time,
                num_chunks_retrieved=num_chunks,
                source_match=source_match,
            )

        except Exception as e:
            print(f"Error evaluating {question_id}: {e}")
            return EvalResult(
                question_id=question_id,
                question=question,
                expected_answer=expected,
                generated_answer=f"ERROR: {str(e)}",
                is_correct=False,
                retrieval_hit=False,
                f1_score=0.0,
                response_time_ms=(time.time() - start_time) * 1000,
                num_chunks_retrieved=0,
                source_match=False,
            )

    async def run_evaluation(
        self,
        csv_path: str,
        max_questions: Optional[int] = None,
    ) -> EvalSummary:
        """Run full evaluation."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from app.models import HuggingFaceProvider

        print(f"\n{'='*60}")
        print("TPN RAG FULL EVALUATION")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Embeddings: {self.embedding_model}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Load data
        print("Loading test data...")
        df = self.load_mcq_data(csv_path)
        if max_questions:
            df = df.head(max_questions)
        print(f"Loaded {len(df)} questions")

        # Initialize components
        print("\nInitializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"trust_remote_code": True},
        )

        print("Loading vector store...")
        vectorstore = Chroma(
            collection_name="tpn_documents",
            embedding_function=embeddings,
            persist_directory=str(project_root / "data" / "chromadb"),
        )

        print("Initializing LLM provider (local mode for large models)...")
        llm = HuggingFaceProvider(model_name=self.model_name, use_local=True)

        # Run evaluation
        print(f"\nEvaluating {len(df)} questions...\n")

        for idx, row in df.iterrows():
            # Support both column formats
            if 'correct_answer' in row:
                q_id = str(row.get('question_id', idx + 1))
                question = row['question']
                expected = row['correct_answer']
            else:
                q_id = str(row.get('ID', idx + 1))
                question = row['Question']
                expected = row['Corrrect Option (s)']

            result = await self.evaluate_single(
                question=question,
                expected=expected,
                question_id=q_id,
                vectorstore=vectorstore,
                llm_provider=llm,
            )
            self.results.append(result)

            # Progress
            if (idx + 1) % 5 == 0:
                correct = sum(1 for r in self.results if r.is_correct)
                acc = correct / len(self.results) * 100
                print(f"  Progress: {idx+1}/{len(df)} | Accuracy: {acc:.1f}%")

        total_time = time.time() - start_time

        # Calculate summary
        correct = sum(1 for r in self.results if r.is_correct)
        retrieval_hits = sum(1 for r in self.results if r.retrieval_hit)

        summary = EvalSummary(
            timestamp=self.timestamp,
            model_name=self.model_name,
            embedding_model=self.embedding_model,
            total_questions=len(self.results),
            correct_answers=correct,
            accuracy=correct / len(self.results) * 100 if self.results else 0,
            retrieval_hit_rate=retrieval_hits / len(self.results) * 100 if self.results else 0,
            mean_f1=np.mean([r.f1_score for r in self.results]),
            mean_response_time_ms=np.mean([r.response_time_ms for r in self.results]),
            total_time_seconds=total_time,
            config={
                "embedding_model": self.embedding_model,
                "llm_model": self.model_name,
                "top_k": 10,
            },
        )

        # Save results
        self.save_results(summary)

        # Generate graphs
        if MATPLOTLIB_AVAILABLE:
            self.generate_graphs(summary)

        # Print summary
        self.print_summary(summary)

        return summary

    def save_results(self, summary: EvalSummary):
        """Save results to JSON."""
        results_file = self.output_dir / f"eval_results_{self.timestamp}.json"

        data = {
            "summary": asdict(summary),
            "detailed_results": [asdict(r) for r in self.results],
        }

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def generate_graphs(self, summary: EvalSummary):
        """Generate academic-style graphs."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Set academic style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

        # Figure 1: Accuracy Bar Chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        metrics = ['Accuracy', 'Retrieval Hit Rate', 'Mean F1 Score']
        values = [summary.accuracy, summary.retrieval_hit_rate, summary.mean_f1 * 100]
        colors = ['#2ecc71', '#3498db', '#9b59b6']

        bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Score (%)', fontweight='bold')
        ax1.set_title(f'TPN RAG Evaluation Results\n{summary.model_name}', fontweight='bold')
        ax1.set_ylim(0, 100)

        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fig1.savefig(self.output_dir / f'accuracy_chart_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: F1 Score Distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        f1_scores = [r.f1_score for r in self.results]
        ax2.hist(f1_scores, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
        ax2.set_xlabel('F1 Score', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution of F1 Scores', fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        fig2.savefig(self.output_dir / f'f1_distribution_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # Figure 3: Response Time Distribution
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        response_times = [r.response_time_ms for r in self.results]
        ax3.hist(response_times, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(response_times), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(response_times):.0f}ms')
        ax3.set_xlabel('Response Time (ms)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Response Time Distribution', fontweight='bold')
        ax3.legend()

        plt.tight_layout()
        fig3.savefig(self.output_dir / f'response_time_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Figure 4: Correct vs Incorrect Pie Chart
        fig4, ax4 = plt.subplots(figsize=(8, 8))

        correct = sum(1 for r in self.results if r.is_correct)
        incorrect = len(self.results) - correct

        sizes = [correct, incorrect]
        labels = [f'Correct\n({correct})', f'Incorrect\n({incorrect})']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)

        ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
        ax4.set_title(f'Answer Accuracy Distribution\n(n={len(self.results)})', fontweight='bold')

        plt.tight_layout()
        fig4.savefig(self.output_dir / f'accuracy_pie_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        print(f"Graphs saved to: {self.output_dir}/")

    def print_summary(self, summary: EvalSummary):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {summary.model_name}")
        print(f"Embeddings: {summary.embedding_model}")
        print(f"Questions: {summary.total_questions}")
        print(f"{'='*60}")
        print(f"Accuracy: {summary.accuracy:.1f}% ({summary.correct_answers}/{summary.total_questions})")
        print(f"Retrieval Hit Rate: {summary.retrieval_hit_rate:.1f}%")
        print(f"Mean F1 Score: {summary.mean_f1:.3f}")
        print(f"Mean Response Time: {summary.mean_response_time_ms:.0f}ms")
        print(f"Total Time: {summary.total_time_seconds:.1f}s")
        print(f"{'='*60}")


async def main():
    """Run full evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="TPN RAG Full Evaluation")
    parser.add_argument("-n", "--num-questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--model", type=str, default="chandramax/tpn-gpt-oss-20b", help="LLM model to use")
    parser.add_argument("--embedding", type=str, default="Qwen/Qwen3-Embedding-8B", help="Embedding model")
    parser.add_argument("--csv", type=str, default="eval/tpn_mcq_cleaned.csv", help="Path to MCQ CSV")
    parser.add_argument("--output", type=str, default="eval/results", help="Output directory")
    parser.add_argument("--local", action="store_true", default=True, help="Use local model inference (default for large models)")
    parser.add_argument("--api", action="store_true", help="Use HuggingFace Inference API instead of local")

    args = parser.parse_args()

    # If --api is specified, use API mode; otherwise use local
    use_local = not args.api

    evaluator = FullEvaluator(
        model_name=args.model,
        embedding_model=args.embedding,
        output_dir=args.output,
        use_local=use_local,
    )

    await evaluator.run_evaluation(
        csv_path=args.csv,
        max_questions=args.num_questions,
    )


if __name__ == "__main__":
    asyncio.run(main())
