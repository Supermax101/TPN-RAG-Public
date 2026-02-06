"""
Embedding-indexed few-shot example pool for dynamic example selection.

Uses sentence-transformers to embed the example questions and select the
k most similar examples for a given query at inference time.

Opt-in via ``dynamic_few_shot=True`` in ExperimentConfig.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FewShotPool:
    """
    Maintains a pool of solved MCQ examples, indexed by question embedding.

    Usage::

        from app.prompting.example_data import TPN_EXAMPLE_POOL
        pool = FewShotPool(TPN_EXAMPLE_POOL)
        examples = pool.select("protein requirement for preterm", k=2)
    """

    def __init__(
        self,
        examples: List[dict],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.examples = list(examples)
        self._model_name = model_name
        self._model = None
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_loaded(self) -> None:
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading FewShotPool embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        questions = [ex["question"] for ex in self.examples]
        self._embeddings = self._model.encode(questions, show_progress_bar=False)

    def select(self, question: str, k: int = 2) -> List[dict]:
        """
        Select the *k* most similar examples to *question*.

        Returns list of dicts with "question" and "answer" keys,
        compatible with ``PromptRenderer.render(few_shot_examples=...)``.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._embeddings is not None

        query_emb = self._model.encode([question], show_progress_bar=False)
        # Cosine similarity via dot product (sentence-transformers normalizes by default)
        scores = (query_emb @ self._embeddings.T)[0]
        top_indices = scores.argsort()[-k:][::-1]
        return [self.examples[int(i)] for i in top_indices]
