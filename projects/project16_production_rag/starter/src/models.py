"""
Pydantic data models — typed I/O for every interface boundary.
Keeps all data contracts in one place.
"""
from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class RawDocument(BaseModel):
    """A document as loaded from disk — before chunking."""
    doc_id:   str
    source:   str              # file path or URL
    title:    str
    content:  str
    metadata: dict = Field(default_factory=dict)

class Chunk(BaseModel):
    """A single chunk ready for embedding + storage."""
    chunk_id:  str             # f"{doc_id}::{chunk_index}"
    doc_id:    str
    source:    str
    title:     str
    content:   str
    chunk_idx: int
    metadata:  dict = Field(default_factory=dict)

class IngestionResult(BaseModel):
    """Summary returned after ingesting a batch of documents."""
    source:         str
    docs_loaded:    int
    chunks_created: int
    chunks_stored:  int
    skipped:        int        # duplicates / too-short chunks
    duration_s:     float
    errors:         list[str] = Field(default_factory=list)


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store, annotated with scores."""
    chunk_id:     str
    doc_id:       str
    source:       str
    title:        str
    content:      str
    vector_score: float = 0.0
    bm25_score:   float = 0.0
    rrf_score:    float = 0.0   # reciprocal rank fusion score
    rerank_score: Optional[float] = None

class RetrievalResult(BaseModel):
    """Full output of one retrieval call."""
    query:   str
    chunks:  list[RetrievedChunk]
    mode:    str = "hybrid"     # vector | bm25 | hybrid
    top_k:   int = 6
    latency_ms: float = 0.0


# ── Agent I/O ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming query from the API client."""
    question:  str
    user_id:   str = "anonymous"
    mode:      str = "hybrid"   # vector | bm25 | hybrid
    top_k:     Optional[int] = None
    rerank:    bool = True

class Citation(BaseModel):
    """A source citation included in the response."""
    title:    str
    source:   str
    chunk_id: str
    snippet:  str              # first 150 chars of the chunk

class QueryResponse(BaseModel):
    """Structured response returned to the API client."""
    answer:        str
    citations:     list[Citation]
    agent:         str          # rag | direct
    retrieval:     Optional[RetrievalResult] = None
    latency_ms:    float = 0.0
    request_id:    str = ""
    model:         str = ""


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    """A single golden evaluation case."""
    case_id:          str
    question:         str
    expected_keywords: list[str]   # at least one must appear in the answer
    expected_sources:  list[str] = Field(default_factory=list)  # optional doc titles

class EvalCaseResult(BaseModel):
    """Result for one evaluation case."""
    case_id:          str
    question:         str
    answer:           str
    citations:        list[str]    # source titles
    keyword_hit:      bool
    source_hit:       bool
    faithfulness:     float        # LLM judge 0–1
    relevancy:        float        # LLM judge 0–1
    overall:          float        # mean of above
    passed:           bool

class EvalReport(BaseModel):
    """Full evaluation run report."""
    timestamp:       str = Field(default_factory=lambda: datetime.now().isoformat())
    total_cases:     int
    passed:          int
    failed:          int
    pass_rate:       float
    avg_faithfulness: float
    avg_relevancy:   float
    avg_overall:     float
    gate_threshold:  float
    gate_passed:     bool
    cases:           list[EvalCaseResult]
