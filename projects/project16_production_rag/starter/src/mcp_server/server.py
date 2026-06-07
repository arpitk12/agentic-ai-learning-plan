"""
TODO — Implement the MCP server exposing RAG tools.

MCP (Model Context Protocol) lets LLM clients call your RAG system as a tool.
Use the FastMCP library — it converts Python functions → MCP tools automatically.

Three tools to expose:
  1. search_docs(query, top_k)  → search the knowledge base
  2. ingest_text(text, title)   → add a new document at runtime
  3. get_stats()                → return index statistics

Pattern:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("production-rag")

    @mcp.tool()
    def my_tool(param: str) -> str:
        ...

Run standalone with:  python -m src.mcp_server.server
"""
from __future__ import annotations
import json
from mcp.server.fastmcp import FastMCP
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.ingestion.pipeline import ingest_text as _ingest_text
from src.ingestion.embedder import embed_query

mcp   = FastMCP("production-rag")
_store     = VectorStore()
_retriever = HybridRetriever(_store)


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base for documents relevant to a query.

    TODO 1: Call _retriever.retrieve(query, top_k=top_k, mode="hybrid")
    TODO 2: Convert result.chunks to a list of dicts:
              [{"chunk_id": c.chunk_id, "source": c.source, "text": c.text, "score": c.score}
               for c in result.chunks]
    TODO 3: Return json.dumps({"results": chunks_list}, indent=2)
    """
    raise NotImplementedError


@mcp.tool()
def ingest_text(text: str, title: str = "Untitled") -> str:
    """
    Ingest a new text document into the knowledge base.

    TODO 4: Call _ingest_text(text=text, title=title, source="mcp", store=_store)
    TODO 5: Call _retriever.rebuild_bm25()
    TODO 6: Return json.dumps({"chunks_created": result.chunks_created,
                                "documents_processed": result.documents_processed})
    """
    raise NotImplementedError


@mcp.tool()
def get_stats() -> str:
    """
    Return knowledge base statistics.

    TODO 7: Call _store.stats()
    TODO 8: Return json.dumps(stats_dict, indent=2)
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Run as standalone MCP server (stdio transport for Claude Desktop / MCP clients)
    mcp.run()
