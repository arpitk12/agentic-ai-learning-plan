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

# TODO: Import the necessary LangGraph components and ChatLiteLLM


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def build_persistent_graph():
    """Build a ReAct graph that persists conversation state across calls."""
    raise NotImplementedError


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
