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
import operator
from typing import TypedDict, Annotated

# TODO: Import StateGraph, END from langgraph.graph
# TODO: Import ChatLiteLLM from langchain_community.chat_models
# TODO: Import or define your tools


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int


# --- Nodes ---
def llm_node(state: AgentState) -> dict:
    """Call the LLM with tools bound and return the updated state."""
    raise NotImplementedError


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls present in the last message."""
    raise NotImplementedError


def should_continue(state: AgentState) -> str:
    """Return 'tools' or 'end' based on whether the last message has tool calls."""
    raise NotImplementedError


# --- Graph ---
def build_graph():
    # TODO: Construct and compile the StateGraph
    raise NotImplementedError


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "messages": [("human", "What is 2^20 and what's today's date?")],
        "step_count": 0,
    })
    print(f"Steps taken: {result['step_count']}")
    print(f"Final answer: {result['messages'][-1].content}")
