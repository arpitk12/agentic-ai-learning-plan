"""
Exercise 1: Build a ReAct Agent with LangGraph
Goal: Recreate the Week 2 ReAct loop using LangGraph's StateGraph.

Tasks:
  1. Define an AgentState TypedDict with: messages, step_count.
  2. Create two nodes: `llm_node` (calls Claude) and `tool_node` (executes tools).
  3. Add a conditional edge from llm_node: if tool_use → tool_node, else → END.
  4. Add a fixed edge tool_node → llm_node.
  5. Compile and invoke with a test question.
  6. Print the full state after execution.

Install: pip install langgraph langchain-community
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import operator
from typing import TypedDict, Annotated
import math
import datetime
# TODO: Import StateGraph, END from langgraph.graph
from langgraph.graph import StateGraph,END
# TODO: Import ChatLiteLLM from langchain_community.chat_models
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from llm import MODEL,tool_result_message
# TODO: Import or define your tools
# --- Tools ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports sqrt, sin, cos, etc."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
    return str(result)


@tool
def get_datetime() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [calculator, get_datetime]
TOOL_MAP = {t.name: t for t in TOOLS}

llm=ChatLiteLLM(model=MODEL,tool=TOOLS)
# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int


# --- Nodes ---
def llm_node(state: AgentState) -> dict:
    """Call the LLM with tools bound and return the updated state."""
    response=llm.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}
    


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls present in the last message."""
    last_msg=state["messages"][-1]
    tool_messages = []
    for tool_call in last_msg.tool_calls:
        fn = TOOL_MAP[tool_call["name"]]
        result = fn.invoke(tool_call["args"])
        tool_messages.append(tool_result_message(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Return 'tools' or 'end' based on whether the last message has tool calls."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# --- Graph ---
def build_graph():
    # TODO: Construct and compile the StateGraph
    graph=StateGraph(AgentState)
    graph.add_node("llm",llm_node)
    graph.add_node("tools",tool_node)
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools","llm")
    graph.set_entry_point("llm")
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "messages": [("human", "What is 2^20 and what's today's date?")],
        "step_count": 0,
    })
    print(f"Steps taken: {result['step_count']}")
    print(f"Final answer: {result['messages'][-1].content}")
