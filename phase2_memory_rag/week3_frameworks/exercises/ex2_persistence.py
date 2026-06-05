"""
Exercise 2: Persistent Memory with LangGraph Checkpointer
Goal: Add conversation persistence so the agent remembers between sessions.

Tasks:
  1. Add a MemorySaver checkpointer to the graph from ex1.
  2. Use a thread_id config so each "user" has their own conversation.
  3. Run two turns with the SAME thread_id and verify the agent remembers turn 1.
  4. Run with a DIFFERENT thread_id and verify it starts fresh.
  5. Inspect the checkpoint store to see what was saved.

Key insight: Checkpointers make agents stateful across calls without
             you writing any custom storage code.

Install: pip install langgraph langchain-community
"""
import operator
from typing import TypedDict, Annotated
from langchain_core.tools import tool
import math
import datetime
from langgraph.graph import StateGraph,END
from langchain_litellm import ChatLiteLLM
from llm import MODEL,tool_result_message
from langgraph.checkpoint.memory import MemorySaver

# TODO: Import the necessary LangGraph components and ChatLiteLLM
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

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def llm_node(state: AgentState) -> dict:
    """Call the LLM with tools bound and return the updated state."""
    response=llm.invoke(state["messages"])
    return {"messages": [response]}
    


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

def build_persistent_graph():
    """Build a ReAct graph that persists conversation state across calls."""
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

    # Turn 1 — Alice
    r1 = app.invoke({"messages": [("human", "My favorite number is 42. Remember that.")]}, config=user_a)
    print("Alice turn 1:", r1["messages"][-1].content)

    # Turn 2 — Alice (should remember 42)
    r2 = app.invoke({"messages": [("human", "What is my favorite number?")]}, config=user_a)
    print("Alice turn 2:", r2["messages"][-1].content)

    # Bob — fresh session (should NOT know about 42)
    r3 = app.invoke({"messages": [("human", "What is my favorite number?")]}, config=user_b)
    print("Bob:", r3["messages"][-1].content)
