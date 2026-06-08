"""
NLI-based faithfulness checker — the core zero-hallucination mechanism.

Algorithm
─────────
  For each sentence in the LLM's generated answer:
    1. Build premise = top-5 retrieved chunks concatenated (up to 2000 chars)
    2. Run (premise, sentence) through cross-encoder/nli-deberta-v3-base
    3. Apply softmax to raw logits → probabilities
    4. Labels: [contradiction=0, entailment=1, neutral=2]  (model-specific order)
    5. If P(entailment) ≥ FAITHFULNESS_THRESHOLD → sentence is grounded ✓
       Else → sentence is not grounded ✗

  faithfulness_score = n_grounded / n_total_sentences
  passed = faithfulness_score ≥ OVERALL_FAITHFULNESS_THRESHOLD

Why DeBERTa?
────────────
  cross-encoder/nli-deberta-v3-base achieves 91.5 on MultiNLI (SOTA range).
  It attends to the full (premise, hypothesis) pair — unlike bi-encoders that
  encode each independently — so it detects subtle contradictions.

  Model size: ~750 MB. Download once, cached in ~/.cache/huggingface/.
  Inference: ~16ms per sentence pair on CPU. For 5 sentences: ~80ms.
  GPU: ~2ms per pair if available.

Label order verification
────────────────────────
  The label order (contradiction=0, entailment=1, neutral=2) is specific to
  this model checkpoint. If you switch models, verify with:
    model.config.id2label  → {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import spacy
from scipy.special import softmax
from sentence_transformers import CrossEncoder

from src.config import cfg
from src.models import FaithfulnessResult, SentenceFaithfulness

logger = logging.getLogger(__name__)

# Label index for cross-encoder/nli-deberta-v3-base
_ENTAILMENT_IDX = 1
_MAX_PREMISE_CHARS = 2000
_MAX_PREMISE_CHUNKS = 5


class FaithfulnessChecker:
    """
    Post-hoc NLI faithfulness checker.

    Load once at API startup via app.state.faithfulness_checker.
    Thread-safe for concurrent query serving (CrossEncoder inference is stateless).
    """

    def __init__(
        self,
        model_name: str = cfg.nli_model,
        per_sentence_threshold: float = cfg.faithfulness_threshold,
        overall_threshold: float = cfg.overall_faithfulness_threshold,
    ) -> None:
        logger.info("Loading NLI model '%s' — first run downloads ~750 MB", model_name)
        self._model = CrossEncoder(model_name, num_labels=3)
        self._per_threshold = per_sentence_threshold
        self._overall_threshold = overall_threshold
        self._nlp = spacy.load("en_core_web_sm")
        logger.info("FaithfulnessChecker ready (per_threshold=%.2f, overall=%.2f)",
                    per_sentence_threshold, overall_threshold)

    def check(
        self,
        answer: str,
        context_chunks: List[dict],
    ) -> FaithfulnessResult:
        """
        Check whether every sentence in `answer` is entailed by `context_chunks`.

        Args:
            answer: Raw LLM-generated answer text.
            context_chunks: List of retrieved chunk dicts (must have 'text' key).

        Returns:
            FaithfulnessResult with per-sentence scores and overall faithfulness.
        """
        if not answer.strip():
            return FaithfulnessResult(
                sentences=[], faithfulness_score=0.0, passed=False, grounded_answer=""
            )

        premise = self._build_premise(context_chunks)
        sentences = self._split_sentences(answer)

        if not sentences:
            return FaithfulnessResult(
                sentences=[], faithfulness_score=0.0, passed=False, grounded_answer=""
            )

        # Batch inference: one forward pass for all sentences
        pairs = [(premise, sent) for sent in sentences]
        raw_logits = self._model.predict(pairs)  # shape: (n_sentences, 3)

        sentence_results: List[SentenceFaithfulness] = []
        for sent, logits in zip(sentences, raw_logits):
            probs = softmax(logits)
            entailment_score = float(probs[_ENTAILMENT_IDX])
            is_grounded = entailment_score >= self._per_threshold

            sentence_results.append(
                SentenceFaithfulness(
                    sentence=sent,
                    entailment_score=entailment_score,
                    is_grounded=is_grounded,
                )
            )
            logger.debug(
                "Sentence [%s...] → entailment=%.3f grounded=%s",
                sent[:60], entailment_score, is_grounded,
            )

        grounded = [s for s in sentence_results if s.is_grounded]
        faithfulness_score = len(grounded) / len(sentence_results)
        grounded_answer = " ".join(s.sentence for s in grounded)

        result = FaithfulnessResult(
            sentences=sentence_results,
            faithfulness_score=faithfulness_score,
            passed=faithfulness_score >= self._overall_threshold,
            grounded_answer=grounded_answer,
        )

        logger.info(
            "Faithfulness check: score=%.3f (%d/%d grounded) passed=%s",
            faithfulness_score, len(grounded), len(sentence_results), result.passed,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_premise(self, chunks: List[dict]) -> str:
        """Concatenate top chunks into premise; truncate to avoid token overflow."""
        texts = [c["text"] for c in chunks[:_MAX_PREMISE_CHUNKS]]
        combined = " ".join(texts)
        return combined[:_MAX_PREMISE_CHARS]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy's rule-based sentenciser."""
        doc = self._nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
