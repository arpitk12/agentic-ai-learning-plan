"""
Tests for the zero-hallucination layer.

These are given complete — run them as you implement each class to verify your work.
  pytest tests/test_hallucination.py -v
"""
import pytest
import numpy as np


# ── FaithfulnessChecker ────────────────────────────────────────────────────

class MockNLIModel:
    """Deterministic mock — always returns high entailment for matching content."""
    def predict(self, pairs):
        results = []
        for premise, hypothesis in pairs:
            if any(word in premise.lower() for word in hypothesis.lower().split()[:3]):
                results.append([0.01, 0.95, 0.04])  # high entailment
            else:
                results.append([0.03, 0.12, 0.85])  # neutral
        return results


class MockSpacy:
    class Sent:
        def __init__(self, text): self.text = text
    class Doc:
        def __init__(self, text):
            self.sents = [MockSpacy.Sent(s.strip()) for s in text.split(".") if s.strip()]
    def __call__(self, text): return self.Doc(text)
    def load(self, name): return self


def make_checker(threshold=0.75, overall=0.80):
    from src.hallucination.faithfulness_checker import FaithfulnessChecker
    checker = FaithfulnessChecker.__new__(FaithfulnessChecker)
    checker._model = MockNLIModel()
    checker._per_threshold = threshold
    checker._overall_threshold = overall
    checker._nlp = MockSpacy()
    return checker


def test_faithfulness_all_grounded():
    checker = make_checker()
    chunks = [{"text": "The API rate limit is 100 requests per minute per user."}]
    answer = "The API rate limit is 100 requests per minute. Users should track their usage."
    result = checker.check(answer, chunks)
    assert result.faithfulness_score > 0.0
    assert len(result.sentences) > 0


def test_faithfulness_empty_answer():
    checker = make_checker()
    result = checker.check("", [{"text": "some context"}])
    assert result.faithfulness_score == 0.0
    assert not result.passed
    assert result.grounded_answer == ""


def test_faithfulness_grounded_answer_excludes_unsupported():
    checker = make_checker(threshold=0.80)
    chunks = [{"text": "The API rate limit is 100 requests per minute."}]
    # Mock will score "quantum physics" low (no overlap with premise)
    answer = "The API limit is 100 requests. Quantum physics is fascinating."
    result = checker.check(answer, chunks)
    assert result.grounded_answer != answer


# ── AbstainPolicy ──────────────────────────────────────────────────────────

def make_faith_result(score, passed, grounded="some answer"):
    from src.models import FaithfulnessResult
    return FaithfulnessResult(sentences=[], faithfulness_score=score, passed=passed, grounded_answer=grounded)


def test_abstain_no_relevant_docs():
    from src.hallucination.abstain_policy import AbstainPolicy, REASON_NO_DOCS
    policy = AbstainPolicy(min_retrieval_score=0.65)
    should, reason = policy.should_abstain(0.50, make_faith_result(0.95, True))
    assert should is True
    assert reason == REASON_NO_DOCS


def test_abstain_low_faithfulness():
    from src.hallucination.abstain_policy import AbstainPolicy, REASON_LOW_FAITH
    policy = AbstainPolicy(min_retrieval_score=0.65)
    should, reason = policy.should_abstain(0.80, make_faith_result(0.50, False))
    assert should is True
    assert reason == REASON_LOW_FAITH


def test_abstain_all_ungrounded():
    from src.hallucination.abstain_policy import AbstainPolicy, REASON_ALL_UNGROUNDED
    policy = AbstainPolicy(min_retrieval_score=0.65)
    should, reason = policy.should_abstain(0.80, make_faith_result(0.0, True, grounded=""))
    assert should is True
    assert reason == REASON_ALL_UNGROUNDED


def test_no_abstain_when_all_pass():
    from src.hallucination.abstain_policy import AbstainPolicy
    policy = AbstainPolicy(min_retrieval_score=0.65)
    should, reason = policy.should_abstain(0.85, make_faith_result(0.90, True, "good answer"))
    assert should is False
    assert reason == ""


# ── SemanticCache ──────────────────────────────────────────────────────────

def test_semantic_cache_hit():
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.97, max_size=100)
    emb = np.random.rand(384).astype(np.float32)
    cache.set(emb, '{"answer": "cached"}')
    result = cache.get(emb)
    assert result is not None


def test_semantic_cache_miss_different_embedding():
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.97, max_size=100)
    emb1 = np.ones(384, dtype=np.float32)
    emb2 = np.zeros(384, dtype=np.float32)
    emb2[0] = 1.0
    cache.set(emb1, '{"answer": "first"}')
    result = cache.get(emb2)
    assert result is None


def test_semantic_cache_lru_eviction():
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.99, max_size=3)
    for i in range(4):
        emb = np.zeros(384, dtype=np.float32)
        emb[i] = 1.0
        cache.set(emb, f'{{"i": {i}}}')
    assert cache.size == 3


def test_semantic_cache_hit_rate():
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache(threshold=0.97, max_size=100)
    emb = np.ones(384, dtype=np.float32)
    cache.set(emb, '{}')
    cache.get(emb)   # hit
    cache.get(np.zeros(384, dtype=np.float32))  # miss
    assert 0.4 < cache.hit_rate < 0.6
