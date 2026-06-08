"""
Citation verifier — maps each grounded sentence to the chunk that best supports it.
"""
from __future__ import annotations

import logging
from typing import List

from scipy.special import softmax
from sentence_transformers import CrossEncoder

from src.config import cfg
from src.models import Citation

logger = logging.getLogger(__name__)

_ENTAILMENT_IDX = 1
_MAX_CHUNK_PREVIEW = 300


class CitationVerifier:
    def __init__(self, nli_model: CrossEncoder) -> None:
        """
        TODO 1: Store nli_model as self._model.
                (Injected from FaithfulnessChecker to avoid loading twice.)
        """
        raise NotImplementedError

    def verify(
        self,
        grounded_answer: str,
        chunks: List[dict],
    ) -> List[Citation]:
        """
        TODO 2: Split grounded_answer into sentences using spaCy.
                Return [] if no sentences or no chunks.

        TODO 3: For each sentence:
                  - Build pairs = [(c["text"][:2000], sentence) for c in chunks]
                  - Call self._model.predict(pairs) → raw logits (n_chunks × 3)
                  - Apply softmax to each row → entailment_scores
                  - Find best_idx = argmax of entailment_scores
                  - Append Citation(sentence, chunk_id, document_title, source, chunk_text[:300])

        TODO 4: Return the list of citations.
        """
        raise NotImplementedError
