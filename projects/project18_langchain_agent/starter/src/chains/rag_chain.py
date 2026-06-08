"""Starter stub — Project 18: LangChain Research Agent.

Complete each TODO to build the LCEL RAG chain and structured output chain.
Run tests with: pytest tests/
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field


# ── Pydantic output schema ────────────────────────────────────────────────────

class ResearchSummary(BaseModel):
    title: str = Field(description="Article title")
    summary: str = Field(description="200-word summary")
    key_points: list[str] = Field(description="5 key takeaways")
    sources: list[str] = Field(description="List of sources")


# ── RAG prompt ───────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below. If unsure, say so.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Join retrieved documents into a single context string."""
    # TODO 1: Join docs using doc.page_content, separated by "\n\n"
    raise NotImplementedError


def build_rag_chain(retriever, llm):
    """Build an LCEL RAG chain: retrieve → format → prompt → LLM → parse.

    Use RunnableParallel to fetch context and pass through the question simultaneously.

    Expected chain shape:
        RunnableParallel(context=..., question=...) | RAG_PROMPT | llm | StrOutputParser()
    """
    # TODO 2: Create a RunnableParallel that:
    #   - 'context': pipes retriever → format_docs
    #   - 'question': uses RunnablePassthrough()
    # TODO 3: Pipe the parallel step → RAG_PROMPT → llm → StrOutputParser()
    raise NotImplementedError


def build_structured_chain(llm):
    """Build a chain that returns a ResearchSummary Pydantic object.

    Use llm.with_structured_output(ResearchSummary).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a structured research summary from the following text."),
        ("human", "{text}"),
    ])
    # TODO 4: Create structured_llm using llm.with_structured_output(ResearchSummary)
    # TODO 5: Return prompt | structured_llm
    raise NotImplementedError
