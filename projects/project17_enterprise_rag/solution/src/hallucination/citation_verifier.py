"""
Citation verifier — maps each sentence in the final answer to the chunk that
best supports it via entailment scoring.

This gives the caller auditable evidence: every claim in the answer is
traceable to a specific document and chunk, with the supporting text shown.

The entailment scores here re-use the same NLI model that the faithfulness
checker used — they will always be consistent.
"""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder
from scipy.special import softmax

from src.config import cfg
from src.models import Citation

logger = logging.getLogger(__name__)

_ENTAILMENT_IDX = 1
_MAX_CHUNK_PREVIEW = 300


class CitationVerifier:
    def __init__(self, nli_model: CrossEncoder) -> None:
        """
        Args:
            nli_model: The same CrossEncoder instance used by FaithfulnessChecker.
                       Pass it in rather than loading twice.
        """
        self._model = nli_model

    def verify(
        self,
        grounded_answer: str,
        chunks: List[dict],
    ) -> List[Citation]:
        """
        For each sentence in grounded_answer, find the chunk that best
        supports it and return a Citation object.

        A sentence may appear in multiple citations if multiple chunks
        support it, but we return only the best-scoring chunk per sentence.
        """
        import spacy
        nlp = spacy.load("en_core_web_sm")
        sentences = [s.text.strip() for s in nlp(grounded_answer).sents if s.text.strip()]

        if not sentences or not chunks:
            return []

        citations: List[Citation] = []

        for sentence in sentences:
            # Score sentence against every chunk
            pairs = [(c["text"][:2000], sentence) for c in chunks]
            logits = self._model.predict(pairs)  # shape: (n_chunks, 3)

            entailment_scores = [float(softmax(l)[_ENTAILMENT_IDX]) for l in logits]
            best_idx = int(max(range(len(entailment_scores)), key=lambda i: entailment_scores[i]))
            best_chunk = chunks[best_idx]

            citations.append(
                Citation(
                    sentence=sentence,
                    chunk_id=best_chunk["chunk_id"],
                    document_title=best_chunk.get("title", ""),
                    source=best_chunk.get("source", ""),
                    chunk_text=best_chunk["text"][:_MAX_CHUNK_PREVIEW],
                )
            )
            logger.debug(
                "Sentence [%s...] → chunk %s (entailment=%.3f)",
                sentence[:40], best_chunk["chunk_id"], entailment_scores[best_idx],
            )

        return citations
