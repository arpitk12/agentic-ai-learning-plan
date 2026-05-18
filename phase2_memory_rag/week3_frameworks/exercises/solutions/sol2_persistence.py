"""
SOLUTION — Exercise 2: Persistent Memory with LangGraph Checkpointer
"""
import math
import datetime
import operator
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307


@tool
def get_datetime() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [calculator, get_datetime]
TOOL_MAP = {t.name: t for t in TOOLS}
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def llm_node(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def tool_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    results = []
    for call in last_msg.tool_calls:
        result = TOOL_MAP[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_persistent_graph():
    memory = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")
    return graph.compile(checkpointer=memory)


if __name__ == "__main__":
    app = build_persistent_graph()

    user_a = {"configurable": {"thread_id": "user-alice"}}
    user_b = {"configurable": {"thread_id": "user-bob"}}

    r1 = app.invoke({"messages": [HumanMessage("My favorite number is 42. Remember that.")]}, config=user_a)
    print("Alice turn 1:", r1["messages"][-1].content)

    r2 = app.invoke({"messages": [HumanMessage("What is my favorite number?")]}, config=user_a)
    print("Alice turn 2:", r2["messages"][-1].content)

    r3 = app.invoke({"messages": [HumanMessage("What is my favorite number?")]}, config=user_b)
    print("Bob:", r3["messages"][-1].content)
