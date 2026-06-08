"""ReAct agent with memory and streaming."""
from __future__ import annotations

import asyncio
import logging

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory

from src.chains.rag_chain import get_llm, get_retriever
from src.config import cfg
from src.tools.tools import calculator, wikipedia_search, get_tavily_tool, get_rag_tool
from src.observability.callbacks import LoggingCallbackHandler

logger = logging.getLogger(__name__)


def build_agent() -> AgentExecutor:
    llm = get_llm()
    retriever = get_retriever()

    tools = [
        get_tavily_tool(cfg.tavily_api_key),
        wikipedia_search,
        calculator,
        get_rag_tool(retriever),
    ]

    # Pull the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        callbacks=[LoggingCallbackHandler()],
    )


async def stream_agent_response(executor: AgentExecutor, question: str) -> None:
    """Stream tokens and tool events using astream_events."""
    print(f"\n🤔 Question: {question}\n")
    async for event in executor.astream_events({"input": question}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n\n🔧 [{event['name']}] ← {str(event['data'].get('input', ''))[:80]}")
        elif kind == "on_tool_end":
            result = str(event["data"].get("output", ""))[:120]
            print(f"   Result: {result}...")
    print("\n")


def run_interactive():
    executor = build_agent()
    print("Research Agent ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        asyncio.run(stream_agent_response(executor, question))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_interactive()
