"""Memory helpers for project 18 — LangChain Research Agent."""
from __future__ import annotations

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.memory.chat_memory import BaseChatMemory


def get_window_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """Return a sliding-window conversation memory (last k turns).

    This is the cheapest form of memory — no LLM calls required.
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
    )


def get_summary_memory(llm) -> ConversationSummaryMemory:
    """Return a summary-based conversation memory.

    Periodically condenses older conversation turns using the LLM.
    Trades LLM tokens for unlimited effective history length.
    """
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
    )
