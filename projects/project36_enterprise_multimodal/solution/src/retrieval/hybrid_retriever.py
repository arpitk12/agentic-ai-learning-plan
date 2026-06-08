"""
solution/src/retrieval/hybrid_retriever.py — Full implementation.
"""
from __future__ import annotations
import asyncio
from src.retrieval.vector_store import query_collection  # type: ignore
from src.graph.neo4j_store import nl_to_cypher, run_query  # type: ignore


async def vector_search(
    question: str, collections: dict, top_k: int = 5
) -> dict[str, list[dict]]:
    async def _query(name: str, col):
        results = await asyncio.to_thread(query_collection, col, question, top_k)
        return name, results

    pairs = await asyncio.gather(
        _query("text",   collections["text"]),
        _query("images", collections["images"]),
        _query("audio",  collections["audio"]),
    )
    return {name: results for name, results in pairs}


async def graph_search(question: str, driver, schema: str) -> list[dict]:
    try:
        cypher = await nl_to_cypher(question, schema)
        rows = await asyncio.to_thread(run_query, driver, cypher)
        facts = []
        for row in rows:
            # Format row as readable fact string
            parts = []
            for k, v in row.items():
                parts.append(f"{k}={v}")
            facts.append({"fact": " | ".join(parts), "cypher": cypher})
        return facts
    except Exception:
        return []


async def hybrid_search(
    question: str,
    collections: dict,
    driver,
    schema: str,
    top_k: int = 5,
    include_graph: bool = True,
) -> dict:
    tasks = [vector_search(question, collections, top_k)]
    if include_graph and driver:
        tasks.append(graph_search(question, driver, schema))

    results = await asyncio.gather(*tasks)
    vec_results = results[0]
    graph_results = results[1] if include_graph and len(results) > 1 else []

    return {
        "text":   sorted(vec_results.get("text", []),   key=lambda x: x["score"], reverse=True)[:top_k],
        "images": sorted(vec_results.get("images", []), key=lambda x: x["score"], reverse=True)[:top_k],
        "audio":  sorted(vec_results.get("audio", []),  key=lambda x: x["score"], reverse=True)[:top_k],
        "graph":  graph_results,
        "total":  (len(vec_results.get("text", [])) +
                   len(vec_results.get("images", [])) +
                   len(vec_results.get("audio", [])) +
                   len(graph_results)),
    }
