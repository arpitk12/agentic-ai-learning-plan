"""
MCP server — exposes RAG capabilities as structured MCP tools.
Run standalone or let the FastAPI lifespan start it as a subprocess.

Tools exposed:
  search_docs(query, top_k)  — retrieve + rerank from the knowledge base
  ingest_text(text, title)   — add content to the knowledge base at runtime
  get_stats()                — collection statistics
"""
from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from src.store.chroma_store import VectorStore
from src.ingestion.embedder import embed_query
from src.ingestion.pipeline import ingest_text as _ingest_text

mcp   = FastMCP("Production RAG MCP Server")
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base and return the top relevant document chunks.
    Use this tool to answer questions grounded in the indexed documents.
    """
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever(_get_store())
    result    = retriever.retrieve(query, top_k=top_k, mode="hybrid")
    if not result.chunks:
        return "No relevant documents found."
    lines = []
    for i, c in enumerate(result.chunks, 1):
        lines.append(f"{i}. [{c.title}] (score={c.rrf_score:.3f})\n   {c.content[:400]}")
    return "\n\n".join(lines)


@mcp.tool()
def ingest_text(text: str, title: str = "dynamic") -> str:
    """
    Add new text content to the knowledge base at runtime.
    The content will be chunked, embedded, and stored immediately.
    """
    result = _ingest_text(text=text, title=title, source="mcp", store=_get_store())
    return (
        f"Ingested '{title}': {result.chunks_stored} chunks stored "
        f"in {result.duration_s}s."
    )


@mcp.tool()
def get_stats() -> str:
    """Return current knowledge base statistics."""
    import json
    return json.dumps(_get_store().stats(), indent=2)


if __name__ == "__main__":
    mcp.run()   # stdio transport
