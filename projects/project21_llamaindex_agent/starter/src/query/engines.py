"""Starter stub — Project 21: SubQuestion and Router query engines."""
from __future__ import annotations


def build_sub_question_engine(indexes: dict):
    """Build a SubQuestionQueryEngine that decomposes multi-part questions.

    Each index becomes a QueryEngineTool with metadata describing its content.
    """
    # TODO 1: from llama_index.core.tools import QueryEngineTool, ToolMetadata
    # TODO 2: from llama_index.core.query_engine import SubQuestionQueryEngine
    # TODO 3: from llama_index.core.postprocessor import SentenceTransformerRerank
    # TODO 4: Build a reranker with model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
    # TODO 5: For each (name, idx) in indexes.items():
    #         Create QueryEngineTool(query_engine=idx.as_query_engine(node_postprocessors=[reranker]),
    #                                metadata=ToolMetadata(name=..., description=...))
    # TODO 6: Return SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, verbose=True)
    raise NotImplementedError


def build_router_engine(vector_index, summary_index):
    """Build a RouterQueryEngine that picks vector or summary engine per query.

    Uses LLMSingleSelector — the LLM reads tool descriptions to pick the right one.
    """
    # TODO 7: from llama_index.core.query_engine import RouterQueryEngine
    # TODO 8: from llama_index.core.selectors import LLMSingleSelector
    # TODO 9: from llama_index.core.tools import QueryEngineTool
    # TODO 10: Create vector_tool (specific facts) and summary_tool (overviews)
    # TODO 11: Return RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), tools=[...])
    raise NotImplementedError
