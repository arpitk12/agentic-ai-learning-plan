"""
Faithfulness checker — NLI-based per-sentence hallucination detection.
★ This is the most important file in the project. Read GUIDE.md Phase 4 first.

Algorithm:
  For each sentence in the LLM answer:
    1. Build premise from top-5 retrieved chunks (max 2000 chars)
    2. Score (premise, sentence) with NLI cross-encoder
    3. Apply softmax → P(entailment)
    4. If P(entailment) ≥ threshold → grounded ✓ else not grounded ✗
  faithfulness_score = n_grounded / n_total
"""
from __future__ import annotations

import logging
from typing import List

from src.config import cfg
from src.models import FaithfulnessResult, SentenceFaithfulness

logger = logging.getLogger(__name__)

_ENTAILMENT_IDX = 1        # label order for cross-encoder/nli-deberta-v3-base
_MAX_PREMISE_CHARS = 2000
_MAX_PREMISE_CHUNKS = 5


class FaithfulnessChecker:
    def __init__(
        self,
        model_name: str = cfg.nli_model,
        per_sentence_threshold: float = cfg.faithfulness_threshold,
        overall_threshold: float = cfg.overall_faithfulness_threshold,
    ) -> None:
        """
        TODO 1: Import CrossEncoder from sentence_transformers.
                Load self._model = CrossEncoder(model_name, num_labels=3).
                Store per_sentence_threshold as self._per_threshold.
                Store overall_threshold as self._overall_threshold.

        TODO 2: Import spacy and load self._nlp = spacy.load("en_core_web_sm").
                This is used for sentence splitting.

        Note: CrossEncoder loading downloads ~750 MB on first run.
        """
        raise NotImplementedError

    def check(
        self,
        answer: str,
        context_chunks: List[dict],
    ) -> FaithfulnessResult:
        """
        TODO 3: Return early if answer is empty/whitespace:
                  FaithfulnessResult(sentences=[], faithfulness_score=0.0, passed=False, grounded_answer="")

        TODO 4: Build premise by calling _build_premise(context_chunks).

        TODO 5: Split answer into sentences by calling _split_sentences(answer).
                Return early (same as TODO 3) if no sentences found.

        TODO 6: Build pairs = [(premise, sent) for sent in sentences].
                Call self._model.predict(pairs) to get raw_logits (shape: n_sentences × 3).

        TODO 7: For each (sentence, logits) pair:
                  - Apply scipy.special.softmax to logits
                  - entailment_score = float(probs[_ENTAILMENT_IDX])
                  - is_grounded = entailment_score >= self._per_threshold
                  - Append SentenceFaithfulness(sentence, entailment_score, is_grounded)

        TODO 8: Compute:
                  - grounded = [s for s in sentence_results if s.is_grounded]
                  - faithfulness_score = len(grounded) / len(sentence_results)
                  - grounded_answer = " ".join(s.sentence for s in grounded)
                  - passed = faithfulness_score >= self._overall_threshold

        TODO 9: Return FaithfulnessResult(sentences, faithfulness_score, passed, grounded_answer).
        """
        raise NotImplementedError

    def _build_premise(self, chunks: List[dict]) -> str:
        """
        TODO 10: Join the 'text' field of the first _MAX_PREMISE_CHUNKS chunks with " ".
                 Truncate to _MAX_PREMISE_CHARS characters.
        """
        raise NotImplementedError

    def _split_sentences(self, text: str) -> List[str]:
        """
        Use self._nlp (spaCy) to split text into sentences.
        Return a list of non-empty stripped sentence strings.
        (This is given — spaCy handles edge cases like abbreviations.)
        """
        doc = self._nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
