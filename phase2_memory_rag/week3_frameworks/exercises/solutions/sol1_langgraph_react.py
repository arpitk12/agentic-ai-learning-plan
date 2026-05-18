"""
SOLUTION — Exercise 1: LangGraph ReAct Agent
"""
import math
import datetime
import operator
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

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

llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(TOOLS)


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int


# --- Nodes ---
def llm_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}


def tool_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_messages = []
    for tool_call in last_msg.tool_calls:
        fn = TOOL_MAP[tool_call["name"]]
        result = fn.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# --- Graph ---
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "messages": [HumanMessage(content="What is 2^20 and what's today's date?")],
        "step_count": 0,
    })
    print(f"Steps taken: {result['step_count']}")
    print(f"Final answer: {result['messages'][-1].content}")
