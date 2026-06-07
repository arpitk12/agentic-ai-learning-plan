"""
MCP server exposing RAG capabilities as tools for LLM clients.

Use the FastMCP library — decorate functions with @mcp.tool() to register them.
Run standalone: python -m src.mcp_server.server
"""
from __future__ import annotations
import json
from mcp.server.fastmcp import FastMCP
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.ingestion.pipeline import ingest_text as _ingest_text

mcp        = FastMCP("production-rag")
_store     = VectorStore()
_retriever = HybridRetriever(_store)


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base for documents relevant to a query.

    TODO 1: Retrieve chunks using the hybrid retriever
    TODO 2: Serialise the chunk list to a JSON string and return it
    """
    raise NotImplementedError


@mcp.tool()
def ingest_text(text: str, title: str = "Untitled") -> str:
    """
    Ingest a new document into the knowledge base.

    TODO 3: Ingest the text and rebuild the BM25 index
    TODO 4: Return a JSON string summarising what was created
    """
    raise NotImplementedError


@mcp.tool()
def get_stats() -> str:
    """TODO 5: Return store statistics as a JSON string."""
    raise NotImplementedError


if __name__ == "__main__":
    mcp.run()
