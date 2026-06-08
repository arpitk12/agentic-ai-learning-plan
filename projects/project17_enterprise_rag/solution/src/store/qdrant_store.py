"""
Qdrant vector store — optimised for 10M+ document chunks.

Scale profile
─────────────
  10M source docs × 5 chunks avg = 50M chunks
  Vector dim:  384  (all-MiniLM-L6-v2)
  float32 RAM: 50M × 384 × 4 bytes = 72 GB
  INT8 RAM:    50M × 384 × 1 byte  = 18 GB  ← fits 24 GB server RAM

HNSW parameters
───────────────
  m=16           — 16 bi-directional links per node; higher = better recall + more RAM
  ef_construct=200 — build-time beam width; higher = better index quality, slower build
  ef=128           — query-time beam width; tune per SLA (higher = more accurate, slower)

Quantization
────────────
  INT8 scalar quantisation clips 1% outliers then maps float32 → int8.
  Recall drop: ~1%.  Mitigate with rescore=True (re-rank candidates with float32).
  oversampling=2.0 fetches 2× more candidates before rescoring.

Payload indexes
───────────────
  Create keyword indexes on "source" and "title" so filtered searches
  do a pre-filter scan, not a full collection scan.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    QuantizationSearchParams,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from src.config import cfg

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(
        self,
        url: str = cfg.qdrant_url,
        collection_name: str = cfg.qdrant_collection,
        vector_size: int = cfg.qdrant_vector_size,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url, timeout=30)

    # ── Collection lifecycle ──────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """Create collection with HNSW + INT8 scalar quantisation."""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection '%s'", self.collection_name)

        if self.client.collection_exists(self.collection_name):
            logger.info("Collection '%s' already exists — skipping creation", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=cfg.qdrant_hnsw_m,
                    ef_construct=cfg.qdrant_hnsw_ef_construct,
                    full_scan_threshold=10_000,
                ),
            ),
            quantization_config=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,      # clip 1% outliers before quantizing
                always_ram=True,    # keep quantized vectors in RAM
            ),
            # Production: shard_number=4, replication_factor=2
            shard_number=1,
            replication_factor=1,
        )

        # Payload indexes — critical for fast filtered search on 50M vectors
        for field, schema in [("source", "keyword"), ("title", "keyword")]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

        logger.info(
            "Created collection '%s' — HNSW(m=%d, ef=%d) + INT8 quantisation",
            self.collection_name, cfg.qdrant_hnsw_m, cfg.qdrant_hnsw_ef_construct,
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert_batch(self, items: List[dict]) -> int:
        """
        Upsert a batch of pre-embedded chunks.

        Each item must have keys: chunk_id, embedding, text, title, source,
        doc_id, chunk_index, metadata.
        Always use batches of 128-512 for maximum throughput.
        """
        points = []
        for item in items:
            # Deterministic UUID from chunk_id so re-ingestion is idempotent
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"]))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=item["embedding"],
                    payload={
                        "doc_id": item["doc_id"],
                        "chunk_id": item["chunk_id"],
                        "chunk_index": item.get("chunk_index", 0),
                        "text": item["text"],
                        "title": item.get("title", ""),
                        "source": item.get("source", ""),
                        "metadata": item.get("metadata", {}),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d vectors", len(points))
        return len(points)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        source_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        """
        Vector search with optional payload filtering.

        Uses rescore=True + oversampling=2.0 to compensate for quantisation
        recall loss: fetches 2× more candidates, rescores with float32, returns top_k.
        """
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=cfg.qdrant_search_ef,
                exact=False,
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,       # re-rank with float32 after int8 ANN
                    oversampling=2.0,   # fetch 2× candidates before rescoring
                ),
            ),
            with_payload=True,
        )

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "doc_id": r.payload["doc_id"],
                "text": r.payload["text"],
                "title": r.payload["title"],
                "source": r.payload["source"],
                "score": float(r.score),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.client.count(self.collection_name).count

    def get_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
