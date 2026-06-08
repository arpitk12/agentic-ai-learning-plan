"""
Pydantic data models — single source of truth for all I/O types.
"""
from __future__ import annotations
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1_000_000)
    title: str = Field(..., min_length=1, max_length=512)
    source: str = Field(default="api", max_length=256)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    topic: str
    partition: int
    offset: int
    status: str = "queued"


# ── Query ─────────────────────────────────────────────────────────────────

class RetrievalMode(str, Enum):
    vector = "vector"
    bm25 = "bm25"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    source_filter: Optional[str] = None   # filter by document source


# ── Faithfulness ──────────────────────────────────────────────────────────

class SentenceFaithfulness(BaseModel):
    sentence: str
    entailment_score: float          # P(entailment) from NLI model
    is_grounded: bool
    supporting_chunk_id: Optional[str] = None


class FaithfulnessResult(BaseModel):
    sentences: List[SentenceFaithfulness]
    faithfulness_score: float        # fraction of grounded sentences
    passed: bool                     # faithfulness_score >= overall_threshold
    grounded_answer: str             # answer with only grounded sentences


# ── Response ──────────────────────────────────────────────────────────────

class Citation(BaseModel):
    sentence: str                    # the answer sentence this citation supports
    chunk_id: str
    document_title: str
    source: str
    chunk_text: str                  # truncated to 300 chars


class QueryResponse(BaseModel):
    answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    faithfulness_score: float = 0.0
    retrieval_score: float = 0.0     # max cosine similarity of top retrieved chunk
    abstained: bool = False
    abstain_reason: Optional[str] = None  # no_relevant_documents | insufficient_grounding
    cached: bool = False
    request_id: str = ""
    latency_ms: float = 0.0


# ── Ops ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    redis: str
    kafka: str


class StatsResponse(BaseModel):
    vectors_count: int
    query_cache_size: int
    semantic_cache_size: int
    embed_cache_hit_rate: float
