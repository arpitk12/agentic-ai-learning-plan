"""
Qdrant vector store — optimised for 10M+ document chunks.

Scale profile (read before implementing):
  10M source docs × 5 chunks avg = 50M chunks
  float32 RAM: 50M × 384 × 4 bytes = 72 GB
  INT8 RAM:    50M × 384 × 1 byte  = 18 GB  ← target with quantization

Implement all TODOs in order. Run the Phase 1 checkpoint from GUIDE.md after each step.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from src.config import cfg

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(
        self,
        url: str = cfg.qdrant_url,
        collection_name: str = cfg.qdrant_collection,
        vector_size: int = cfg.qdrant_vector_size,
    ) -> None:
        """
        TODO 1: Import QdrantClient from qdrant_client and create self.client.
                Use QdrantClient(url=url, timeout=30).
                Store collection_name and vector_size as instance attributes.
        """
        raise NotImplementedError

    def create_collection(self, recreate: bool = False) -> None:
        """
        TODO 2: If recreate=True and the collection already exists, delete it first.

        TODO 3: Skip creation if the collection already exists
                (client.collection_exists(self.collection_name)).

        TODO 4: Call client.create_collection() with:
                  - VectorParams(size=self.vector_size, distance=Distance.COSINE,
                      hnsw_config=HnswConfigDiff(m=cfg.qdrant_hnsw_m,
                                                  ef_construct=cfg.qdrant_hnsw_ef_construct))
                  - quantization_config=ScalarQuantizationConfig(
                        type=ScalarType.INT8, quantile=0.99, always_ram=True)

        TODO 5: Create keyword payload indexes on "source" and "title" fields
                (critical for fast filtered search on 50M vectors).
                Use client.create_payload_index(collection_name, field_name, field_schema).

        Hint — imports needed:
          from qdrant_client.models import (Distance, VectorParams, HnswConfigDiff,
            ScalarQuantizationConfig, ScalarType)
        """
        raise NotImplementedError

    def upsert_batch(self, items: List[dict]) -> int:
        """
        TODO 6: Build a list of PointStruct objects from `items`.
                Each item has keys: chunk_id, embedding, text, title, source, doc_id,
                chunk_index, metadata.
                Use uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"]) as the point ID
                (ensures re-ingestion is idempotent).

        TODO 7: Call client.upsert(collection_name=self.collection_name, points=points)
                and return the count of upserted points.

        Hint: PointStruct(id=..., vector=item["embedding"], payload={...})
        """
        raise NotImplementedError

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        source_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        """
        TODO 8: Build an optional Filter if source_filter is provided.
                Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])

        TODO 9: Call client.search() with:
                  - query_vector, limit=top_k, query_filter, score_threshold
                  - search_params=SearchParams(hnsw_ef=cfg.qdrant_search_ef, exact=False,
                      quantization=QuantizationSearchParams(rescore=True, oversampling=2.0))
                  - with_payload=True

        TODO 10: Convert results to list of dicts with keys:
                 chunk_id, doc_id, text, title, source, score, metadata.
                 Extract values from r.payload and r.score.

        Hint — imports needed:
          from qdrant_client.models import (Filter, FieldCondition, MatchValue,
            SearchParams, QuantizationSearchParams)
        """
        raise NotImplementedError

    def count(self) -> int:
        """TODO 11: Return client.count(self.collection_name).count"""
        raise NotImplementedError

    def get_info(self) -> dict:
        """TODO 12: Return a dict with vectors_count, points_count, status from client.get_collection()."""
        raise NotImplementedError

    def is_healthy(self) -> bool:
        try:
            # TODO 12b: Call client.get_collections() and return True; return False on exception.
            raise NotImplementedError
        except Exception:
            return False
