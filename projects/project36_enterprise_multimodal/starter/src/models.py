"""
src/models.py — Pydantic schemas for all API requests and responses.

TODOs:
  1. Define AnalyzeRequest and AnalyzeResponse
  2. Define IngestResult (returned after ingesting a document)
  3. Define SearchResult (returned by /search endpoint)
  4. Define MemoryEntry (single Mem0 memory record)
  5. Define GuardrailResult (result of the 4-layer pipeline)
  6. Define HealthResponse (circuit breaker states + service pings)
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any


# ── TODO 1: Analyze request/response ─────────────────────────────────────────
# class AnalyzeRequest(BaseModel):
#     user_id: str
#     question: str
#     include_graph: bool = True
#     top_k: int = Field(5, ge=1, le=20)
#
# class AnalyzeResponse(BaseModel):
#     answer: str
#     sources: list[dict]          # [{text, score, source, page}]
#     graph_facts: list[dict]      # [{subject, predicate, object}]
#     memories_used: int
#     model_used: str
#     cost_usd: float
#     latency_ms: int


# ── TODO 2: Ingest result ─────────────────────────────────────────────────────
# class IngestResult(BaseModel):
#     doc_id: str
#     source: str
#     chunks: int
#     images: int
#     entities: int
#     audio_segments: int = 0
#     cost_usd: float


# ── TODO 3: Search result ─────────────────────────────────────────────────────
# class SearchResult(BaseModel):
#     text_hits: list[dict]
#     image_hits: list[dict]
#     audio_hits: list[dict]
#     graph_hits: list[dict]
#     total: int


# ── TODO 4: Memory entry ──────────────────────────────────────────────────────
# class MemoryEntry(BaseModel):
#     id: str
#     text: str
#     memory_type: str   # episodic | semantic | procedural | profile
#     score: float = 0.0
#     created_at: str = ""


# ── TODO 5: Guardrail result ──────────────────────────────────────────────────
# class GuardrailResult(BaseModel):
#     safe: bool
#     sanitized_text: str          # text after PII anonymization
#     issues: list[str]            # ["injection_detected", "pii_found: email"]
#     pii_types_found: list[str]   # ["email", "phone"]
#     blocked_layer: str | None = None   # "L1" | "L2" | "L3" | "L4" | None


# ── TODO 6: Health response ───────────────────────────────────────────────────
# class CircuitState(BaseModel):
#     model: str
#     state: str   # closed | open | half_open
#     failures: int
#
# class HealthResponse(BaseModel):
#     status: str   # healthy | degraded | unhealthy
#     neo4j: bool
#     redis: bool
#     chroma: bool
#     circuits: list[CircuitState]

raise NotImplementedError("Implement Pydantic models in src/models.py")
