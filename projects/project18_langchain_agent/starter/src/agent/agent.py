"""Starter stub — Project 18: ReAct agent + streaming.

Build a ReAct agent with memory and async streaming.
"""
from __future__ import annotations

import asyncio


def build_agent(tools: list, llm):
    """Build a ReAct AgentExecutor with sliding-window memory.

    Steps:
    1. Create a ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    2. Create a ChatPromptTemplate with system message, chat_history placeholder, human input, agent_scratchpad
    3. Call create_react_agent(llm, tools, prompt) to get the agent
    4. Wrap in AgentExecutor(agent=..., tools=..., memory=..., verbose=True, handle_parsing_errors=True)
    """
    # TODO 1: Import ConversationBufferWindowMemory from langchain.memory
    # TODO 2: Import create_react_agent from langchain.agents
    # TODO 3: Import AgentExecutor from langchain.agents
    # TODO 4: Build the prompt (use hub.pull("hwchase17/react-chat") or build manually)
    # TODO 5: Create the agent chain and AgentExecutor — return the executor
    raise NotImplementedError


async def stream_agent_response(query: str, agent_executor) -> None:
    """Stream agent events using astream_events (version='v2').

    Print token-by-token output for 'on_chat_model_stream' events,
    and tool start/end info for 'on_tool_start'/'on_tool_end' events.
    """
    # TODO 6: Iterate over agent_executor.astream_events({"input": query}, version="v2")
    # TODO 7: For event["event"] == "on_chat_model_stream": print the token chunk
    # TODO 8: For event["event"] == "on_tool_start": print the tool name and input
    raise NotImplementedError


def run_interactive(agent_executor) -> None:
    """Run a simple REPL loop: read user input → invoke agent → print output."""
    # TODO 9: Loop: input("You: "), call agent_executor.invoke({"input": user_input}),
    #         print result["output"]. Break on "quit"/"exit".
    raise NotImplementedError
