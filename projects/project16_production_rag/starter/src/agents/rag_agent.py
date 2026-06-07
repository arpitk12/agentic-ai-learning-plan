"""
RAG agent: retrieve → rerank → generate with grounded citations.
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import QueryRequest, QueryResponse, Citation, RetrievedChunk
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def rag_agent(
    request: QueryRequest,
    retriever: HybridRetriever,
    request_id: str = "?",
) -> QueryResponse:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    TODO 1: Retrieve relevant chunks for the question using the retriever
    TODO 2: Rerank the retrieved chunks to keep only the most relevant ones
    TODO 3: Build a numbered context string from the reranked chunks, labelling each with its source
    TODO 4: Build a grounded system prompt that tells the LLM to answer only from context and cite sources
    TODO 5: Call the LLM with the system prompt and the user's question
    TODO 6: Extract the answer text from the LLM response
    TODO 7: Build a Citation list from the reranked chunks
    TODO 8: Return a QueryResponse with the answer, citations, retrieval mode, and request id
    """
    raise NotImplementedError
