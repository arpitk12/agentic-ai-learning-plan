"""
src/retrieval/hybrid_retriever.py
Combine vector search (ChromaDB) with graph traversal (Neo4j).

TODOs:
  1. implement vector_search() — query all 3 modality collections concurrently
  2. implement graph_search() — NL → Cypher → Neo4j results
  3. implement hybrid_search() — merge + score both result types, rerank
"""
from __future__ import annotations
import asyncio


# ── TODO 1: Vector search across all modalities ───────────────────────────────
async def vector_search(
    question: str,
    collections: dict,   # {"text": col, "images": col, "audio": col}
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """
    Query all 3 ChromaDB collections concurrently.

    Steps:
      1a. Define a helper: async def _query(name, col) → (name, results)
          Inside: run query_collection(col, question, top_k) in a thread executor
          (ChromaDB is sync — wrap with asyncio.to_thread)
      1b. asyncio.gather(_query("text", ...), _query("images", ...), _query("audio", ...))
      1c. Return {"text": [...], "images": [...], "audio": [...]}
    """
    raise NotImplementedError


# ── TODO 2: Graph search ──────────────────────────────────────────────────────
async def graph_search(
    question: str,
    driver,
    schema: str,
) -> list[dict]:
    """
    Convert question to Cypher and run on Neo4j.

    Steps:
      2a. cypher = await nl_to_cypher(question, schema)
      2b. rows = run_query(driver, cypher)  — wrap in asyncio.to_thread
      2c. Convert Neo4j rows to dicts with {"fact": str, "cypher": cypher}
          Format: "subject → predicate → object"
      2d. Return [] on any exception (don't fail the whole retrieval)
    """
    # from src.graph.neo4j_store import nl_to_cypher, run_query
    raise NotImplementedError


# ── TODO 3: Hybrid search ─────────────────────────────────────────────────────
async def hybrid_search(
    question: str,
    collections: dict,
    driver,
    schema: str,
    top_k: int = 5,
    include_graph: bool = True,
) -> dict:
    """
    Run vector search + graph search concurrently and merge results.

    Steps:
      3a. Run vector_search and (if include_graph) graph_search with asyncio.gather
      3b. vector results: already scored 0-1; graph results: assign score=1.0
          (graph matches are exact — treat as highest confidence)
      3c. Merge all text + image + audio results into one ranked list;
          graph results go into a separate "graph" key
      3d. Re-sort text/image/audio by score descending, keep top_k
      3e. Return:
          {
            "text":   [top_k text hits],
            "images": [top_k image hits],
            "audio":  [top_k audio hits],
            "graph":  [graph facts],
            "total":  int,
          }
    """
    raise NotImplementedError
