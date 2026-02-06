"""
NLI-based citation verification using cross-encoder entailment models.

Uses a Natural Language Inference (NLI) cross-encoder to classify
whether a retrieved chunk *entails*, *contradicts*, or is *neutral*
to an answer sentence. This provides a semantic grounding signal
that complements the heuristic overlap used by CitationGrounder.

Opt-in: pass ``use_nli=True`` to CitationGrounder.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Label order for nli-deberta-v3-base: [contradiction, entailment, neutral]
_LABELS = ["contradiction", "entailment", "neutral"]


class NLIGrounder:
    """
    Verify whether a chunk entails an answer sentence using an NLI model.

    Lazy-loads the cross-encoder on first call to avoid import-time cost.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading NLI model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)

    def verify(self, answer_sentence: str, chunk_content: str) -> Tuple[str, float]:
        """
        Classify the relationship between chunk (premise) and answer (hypothesis).

        Args:
            answer_sentence: The generated answer sentence (hypothesis).
            chunk_content: The retrieved chunk text (premise).

        Returns:
            (label, confidence) where label is one of
            ``"entailment"``, ``"neutral"``, ``"contradiction"``.
        """
        self._ensure_loaded()
        assert self._model is not None

        scores = self._model.predict([(chunk_content, answer_sentence)])
        scores = np.atleast_2d(scores)
        label_idx = int(scores[0].argmax())
        return _LABELS[label_idx], float(scores[0][label_idx])
