"""
Baseline Model Evaluation - Direct Model Testing WITHOUT RAG
Tests models on questions without any document access.
Provides baseline performance for RAG comparison.
"""

import asyncio
import pandas as pd
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import httpx
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.providers.ollama import OllamaLLMProvider
from app.providers.openai import OpenAILLMProvider
from app.providers.xai import XAILLMProvider
from app.providers.gemini import GeminiLLMProvider
from app.providers.kimi import KimiLLMProvider


@dataclass
class BaselineResult:
    """Results for a single baseline question evaluation."""
    question_id: str
    question: str
    correct_answer: str
    model_answer: str
    is_correct: bool
    response_time_ms: float
    model_confidence: str
    raw_response: str


class BaselineModelEvaluator:
    """Evaluates raw model performance on questions WITHOUT any RAG enhancement."""
    
    def __init__(self, csv_path: str, selected_model: str = "mistral:7b", provider: str = "ollama"):
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.provider = provider
        
        if provider == "openai":
            self.llm_provider = OpenAILLMProvider(default_model=selected_model)
        elif provider == "xai":
            self.llm_provider = XAILLMProvider(default_model=selected_model)
        elif provider == "gemini":
            self.llm_provider = GeminiLLMProvider(default_model=selected_model)
        elif provider == "kimi":
            self.llm_provider = KimiLLMProvider(default_model=selected_model)
        else:
            self.llm_provider = OllamaLLMProvider(default_model=selected_model)
        
        self.results: List[BaselineResult] = []
        
        print(f"Loading questions from: {csv_path}")
        self.questions_df = self.load_mcq_questions()
        print(f"Loaded {len(self.questions_df)} MCQ questions")
    
    def load_mcq_questions(self) -> pd.DataFrame:
        """Load MCQ questions from CSV."""
        df = pd.read_csv(self.csv_path, keep_default_na=False)
        
        # Support both old and new column names
        if 'correct_answer' in df.columns:
            # New format
            required = ['question_id', 'question', 'options', 'correct_answer']
        else:
            # Old format
            required = ['Question', 'Options', 'Corrrect Option (s)']
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df
    
    def create_baseline_prompt(self, question: str, options: str, case_context: str = "", answer_type: str = "single") -> str:
        """Create baseline prompt without RAG context - tests raw LLM TPN knowledge."""
        
        type_info = "Select ALL correct answers (comma-separated)" if answer_type == "multi" else "Select the ONE best answer"
        
        system_prompt = f"""You are a TPN (Total Parenteral Nutrition) Clinical Specialist taking a board-style certification examination.

BASELINE EVALUATION - Testing your inherent clinical knowledge (no reference materials provided).

EXAMINATION INSTRUCTIONS:
- Answer based on your knowledge of ASPEN guidelines, neonatal nutrition, and TPN protocols
- Options can be A, B, C, D, E, or F
- {type_info}
- For "FALSE" or "LEAST likely" questions: Identify the incorrect statement

RESPONSE FORMAT:
Start your answer with ONLY the letter(s), then provide brief rationale.
Examples: "A" or "F" or "A,B,C" or "B,D,F"

Your response MUST begin with the answer letter(s)."""
        
        user_prompt = ""
        if case_context and isinstance(case_context, str) and case_context.strip():
            user_prompt += f"\nCLINICAL CASE:\n{case_context}\n"
        
        user_prompt += f"""
BOARD EXAMINATION QUESTION:
{question}

QUESTION TYPE: {type_info}

OPTIONS:
{options}

Your answer:"""
        
        return system_prompt + "\n" + user_prompt
    
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
    
    async def evaluate_single_question(
        self,
        question_id: str,
        question: str,
        options: str,
        correct_option: str,
        case_context: str = ""
    ) -> BaselineResult:
        """Evaluate a single question using direct model inference (no RAG)."""
        
        print(f"\nQuestion {question_id}: {question[:70]}...")
        
        start_time = time.time()
        
        try:
            prompt = self.create_baseline_prompt(question, options, case_context)
            
            raw_response = await self.llm_provider.generate(
                prompt=prompt,
                model=self.selected_model,
                temperature=0.0,
                max_tokens=8000,
                seed=42
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            model_answer = "UNKNOWN"
            confidence = "low"
            
            try:
                cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
                
                json_match = re.search(r'\{[^}]*\}', cleaned_response)
                if json_match:
                    response_json = json.loads(json_match.group())
                    model_answer = response_json.get("answer", "UNKNOWN")
                    confidence = response_json.get("confidence", "low")
                else:
                    letters = re.findall(r'\b([A-F])\b', cleaned_response.upper())
                    if letters:
                        from collections import Counter
                        letter_counts = Counter(letters)
                        model_answer = letter_counts.most_common(1)[0][0]
            except (json.JSONDecodeError, KeyError):
                letters = re.findall(r'\b([A-F])\b', raw_response.upper())
                if letters:
                    from collections import Counter
                    model_answer = Counter(letters).most_common(1)[0][0]
            
            correct_normalized = self.normalize_answer(correct_option)
            model_normalized = self.normalize_answer(model_answer)
            
            is_correct = model_normalized == correct_normalized
            
            result = BaselineResult(
                question_id=question_id,
                question=question,
                correct_answer=correct_normalized,
                model_answer=model_normalized,
                is_correct=is_correct,
                response_time_ms=response_time_ms,
                model_confidence=confidence,
                raw_response=raw_response[:500]
            )
            
            status = "CORRECT" if is_correct else "WRONG"
            print(f"   {status}: Expected '{correct_normalized}' -> Got '{model_normalized}' ({response_time_ms:.0f}ms)")
            
            return result
            
        except Exception as e:
            print(f"   Error: {e}")
            response_time_ms = (time.time() - start_time) * 1000
            
            return BaselineResult(
                question_id=question_id,
                question=question,
                correct_answer=self.normalize_answer(correct_option),
                model_answer="ERROR",
                is_correct=False,
                response_time_ms=response_time_ms,
                model_confidence="low",
                raw_response=f"Error: {str(e)}"
            )
    
    async def run_baseline_evaluation(self, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """Run complete baseline evaluation without RAG system."""
        
        print(f"\nStarting BASELINE evaluation for model: {self.selected_model}")
        print(f"Testing raw model knowledge WITHOUT any document access")
        print("=" * 60)
        
        start_time = time.time()
        
        questions_to_process = self.questions_df.head(max_questions) if max_questions else self.questions_df
        
        for idx, row in questions_to_process.iterrows():
            # Support both old and new column names
            if 'correct_answer' in row:
                # New format
                question_id = str(row.get('question_id', idx + 1))
                question = row['question']
                options = row['options']
                correct_option = row['correct_answer']
                case_context = row.get('case_context', '')
                answer_type = row.get('answer_type', 'single')
            else:
                # Old format
                question_id = str(row.get('ID', idx + 1))
                question = row['Question']
                options = row['Options']
                correct_option = row['Corrrect Option (s)']
                case_context = row.get('Case Context if available', '')
                answer_type = 'single'
            
            if not isinstance(case_context, str):
                case_context = ''
            
            result = await self.evaluate_single_question(
                question_id=question_id,
                question=question,
                options=options,
                correct_option=correct_option,
                case_context=case_context
            )
            self.results.append(result)
            
            if len(self.results) % 5 == 0:
                current_accuracy = sum(r.is_correct for r in self.results) / len(self.results) * 100
                print(f"\nProgress: {len(self.results)}/{len(questions_to_process)} | Accuracy: {current_accuracy:.1f}%")
        
        total_time = time.time() - start_time
        
        evaluation_summary = self.calculate_metrics(total_time)
        self.save_results(evaluation_summary)
        
        return evaluation_summary
    
    def calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        avg_response_time = sum(r.response_time_ms for r in self.results) / total_questions
        
        summary = {
            'model_name': self.selected_model,
            'evaluation_type': 'BASELINE_NO_RAG',
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy_percentage': round(accuracy, 2),
            'avg_response_time_ms': round(avg_response_time, 1),
            'total_evaluation_time_seconds': round(total_time, 1)
        }
        
        print("\n" + "="*60)
        print("BASELINE EVALUATION RESULTS (NO RAG)")
        print("="*60)
        print(f"Model: {self.selected_model}")
        print(f"Accuracy: {accuracy:.1f}% ({correct_answers}/{total_questions})")
        print(f"Avg Response Time: {avg_response_time:.0f}ms")
        print(f"Total Time: {total_time:.1f}s")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = self.selected_model.replace(":", "_")
        
        json_file = f"baseline_{model_clean}_{timestamp}.json"
        json_path = Path("eval") / json_file
        
        json_data = {
            'summary': summary,
            'detailed_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'correct_answer': r.correct_answer,
                    'model_answer': r.model_answer,
                    'is_correct': r.is_correct,
                    'response_time_ms': r.response_time_ms,
                    'confidence': r.model_confidence
                }
                for r in self.results
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nResults saved to: {json_file}")


async def get_available_models():
    """Get all available models."""
    from app.config import settings
    
    all_models = []
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                llm_models = [m for m in models if "embed" not in m.lower()]
                all_models.extend([("ollama", m) for m in llm_models])
    except Exception:
        pass
    
    if settings.openai_api_key:
        all_models.append(("openai", "gpt-4o"))
    if settings.gemini_api_key:
        all_models.extend([("gemini", "gemini-2.5-flash")])
    if settings.xai_api_key:
        all_models.append(("xai", "grok-4-fast-reasoning"))
    if settings.kimi_api_key:
        all_models.append(("kimi", "kimi-k2-0905-preview"))
    
    return all_models


async def main():
    """Main baseline evaluation function."""
    
    print("Baseline Model Evaluation - Raw Model Performance Test")
    print("=" * 60)
    print("Purpose: Test models WITHOUT RAG system to establish baseline")
    print("=" * 60)
    
    csv_path = "eval/tpn_mcq_cleaned.csv"
    
    available_models = await get_available_models()
    if not available_models:
        print("No models available")
        return
    
    print(f"\nAvailable Models ({len(available_models)}):")
    for i, (provider, model) in enumerate(available_models, 1):
        print(f"  {i}. [{provider.upper()}] {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}): ").strip()
            if not choice:
                selected_provider, selected_model = available_models[0]
                break
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                selected_provider, selected_model = available_models[choice_num - 1]
                break
        except ValueError:
            pass
    
    print(f"Selected: [{selected_provider.upper()}] {selected_model}")
    
    max_questions_input = input(f"\nLimit questions? (default: all): ").strip()
    max_questions = int(max_questions_input) if max_questions_input.isdigit() else None
    
    try:
        evaluator = BaselineModelEvaluator(csv_path, selected_model, selected_provider)
        await evaluator.run_baseline_evaluation(max_questions)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
