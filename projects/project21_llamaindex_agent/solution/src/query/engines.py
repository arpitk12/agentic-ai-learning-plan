"""SubQuestion + Router query engines — multi-document intelligent querying."""
from __future__ import annotations

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def build_sub_question_engine(indexes: dict[str, VectorStoreIndex]) -> SubQuestionQueryEngine:
    """Break complex questions into sub-questions, one per document collection.

    Args:
        indexes: Dict mapping doc collection name → VectorStoreIndex.

    Returns:
        SubQuestionQueryEngine that decomposes queries automatically.
    """
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3,
    )

    tools = [
        QueryEngineTool(
            query_engine=idx.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[reranker],
            ),
            metadata=ToolMetadata(
                name=name.replace(" ", "_"),
                description=f"Answers questions about: {name}",
            ),
        )
        for name, idx in indexes.items()
    ]

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        verbose=True,
    )


def build_router_engine(
    vector_index: VectorStoreIndex,
    summary_index: SummaryIndex,
) -> RouterQueryEngine:
    """Route queries to either vector (specific facts) or summary (overviews) engine.

    The LLM selector picks the best engine based on the query type.
    """
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(similarity_top_k=5),
        description="Useful for answering specific factual questions, definitions, and details.",
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
        description="Useful for summarizing large documents or getting high-level overviews.",
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=True,
    )
