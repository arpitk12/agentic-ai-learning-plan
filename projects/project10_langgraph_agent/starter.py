"""
Project 10 Starter — LangGraph Decision Agent (HITL + Checkpointing)

Build a stateful research-and-execute agent using LangGraph's StateGraph:
  1. RESEARCH node  — gather information about the query
  2. PLAN node      — create an action plan + classify risk (LOW/MEDIUM/HIGH)
  3. Conditional routing — HIGH/MEDIUM risk → human approval, LOW → auto-execute
  4. APPROVE node   — human reviews the plan (y/n)
  5. EXECUTE node   — implement the approved plan
  6. Checkpointing  — MemorySaver allows resuming interrupted agents

Install:
    pip install langgraph

Usage:
    python starter.py "Summarise the top 3 multi-agent design patterns"
    python starter.py "Analyse trade-offs of different vector databases"

What you need to implement (TODOs 1-5):
  1. research_node(state)       — search + update research_results
  2. plan_node(state)           — LLM creates plan + risk level
  3. route_after_plan(state)    — routing function: "human_approval" or "execute"
  4. execute_node(state)        — implement the plan using research results
  5. Build the StateGraph       — add_node, add_edge, add_conditional_edges, compile
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import re
from typing import TypedDict, Annotated, Literal
import operator
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("⚠  langgraph not installed. Run: pip install langgraph")


# ── State Schema ───────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:            str
    research_results: Annotated[list[str], operator.add]  # append-only
    plan:             str
    risk_level:       str           # "LOW" | "MEDIUM" | "HIGH"
    approved:         bool
    execution_result: str
    steps:            int


# ── LLM Helper (already complete) ─────────────────────────────────────────────

def llm_call(messages: list[dict], system: str = "", max_tokens: int = 1024) -> str:
    """Synchronous LLM call returning plain text."""
    r = chat(messages, system=system, max_tokens=max_tokens)
    return get_text(r)


# ── Mock Tool (already complete) ───────────────────────────────────────────────

def web_search(query: str) -> str:
    """Simulated web search — returns relevant snippets."""
    snippets = {
        "multi-agent": (
            "Key multi-agent patterns: (1) Orchestrator-Worker — planner delegates to specialists. "
            "(2) Debate/Adversarial — agents argue pro/con, judge decides. "
            "(3) Fan-Out/Fan-In — parallel processing with asyncio.gather(). "
            "(4) Map-Reduce — large dataset processed in parallel batches."
        ),
        "vector database": (
            "Vector DB trade-offs: FAISS (fast, in-memory, no filtering), "
            "ChromaDB (dev-friendly, easy setup, <5M vectors), "
            "Qdrant (production-grade, advanced filtering, billions of vectors), "
            "pgvector (SQL+vector in one DB, good for ACID needs)."
        ),
        "langgraph": (
            "LangGraph uses StateGraph with TypedDict state. Nodes are functions, edges are transitions. "
            "Supports checkpointing (MemorySaver/SqliteSaver), conditional routing, and HITL via interrupt_before."
        ),
    }
    for key, result in snippets.items():
        if key.lower() in query.lower():
            return result
    return f"Research results for '{query}': This topic covers several important considerations including trade-offs, best practices, and production deployment strategies."


# ── Graph Nodes ────────────────────────────────────────────────────────────────

def research_node(state: ResearchState) -> dict:
    """
    Gather information about the query using web search.

    TODO 1:
      a. Print progress: print(f"🔍 Researching: {state['query'][:60]}...")
      b. Perform a web search: result = web_search(state["query"])
      c. Return:
             {
               "research_results": [result],   ← note: list, because it's append-only
               "steps": state["steps"] + 1,
             }

    The Annotated[list, operator.add] type means returning a list here will
    APPEND to the existing list, not replace it.
    """
    # TODO 1: implement research node
    raise NotImplementedError("research_node() not implemented yet")


def plan_node(state: ResearchState) -> dict:
    """
    Create an action plan and classify its risk level.

    TODO 2:
      a. Print: print("📋 Creating plan...")
      b. Build context from research results:
             context = "\\n\\n".join(state["research_results"])
      c. Ask the LLM to create a plan + risk assessment. Return ONLY JSON:
             system = (
                 "You are a strategic planner. Create a concise action plan based on the research. "
                 "Classify risk: LOW (read-only, no external actions), "
                 "MEDIUM (creates content but no external publishing), "
                 "HIGH (sends emails, posts publicly, modifies databases). "
                 'Return ONLY JSON: {"plan": "steps...", "risk_level": "LOW|MEDIUM|HIGH", '
                 '"reasoning": "why this risk level"}'
             )
             messages = [{"role": "user", "content":
                 f"Query: {state['query']}\\n\\nResearch:\\n{context}\\n\\nCreate plan."}]
      d. Parse JSON: s = raw.find("{"); e = raw.rfind("}") + 1; data = json.loads(raw[s:e])
      e. Print: print(f"  Risk: {data['risk_level']} — {data['reasoning'][:80]}")
      f. Return: {"plan": data["plan"], "risk_level": data["risk_level"], "steps": state["steps"]+1}

    On JSON parse failure, return a default LOW-risk plan to avoid blocking the graph.
    """
    # TODO 2: implement plan node
    raise NotImplementedError("plan_node() not implemented yet")


def human_approval_node(state: ResearchState) -> dict:
    """
    Pause for human review of the plan. (Already implemented — study this pattern.)

    This is the HITL node. In a web app you'd store the state and wait for a
    webhook; in a CLI you use input(). LangGraph's interrupt_before can also
    pause the graph before this node fires.
    """
    print(f"\n{'='*55}")
    print(f"⚠️  HUMAN APPROVAL REQUIRED")
    print(f"{'='*55}")
    print(f"Risk level: {state['risk_level'].upper()}")
    print(f"\nProposed plan:\n{state['plan']}")
    print(f"{'='*55}")
    answer = input("Approve this plan? [y/n]: ").strip().lower()
    approved = answer == "y"
    if approved:
        print("✅ Plan approved. Proceeding to execute.")
    else:
        print("❌ Plan rejected by operator.")
    return {"approved": approved, "steps": state["steps"] + 1}


def execute_node(state: ResearchState) -> dict:
    """
    Implement the approved plan using the research context.

    TODO 4:
      a. Print: print("⚙️  Executing plan...")
      b. Build context from the last 2 research results (to fit context window):
             context = "\\n\\n".join(state["research_results"][-2:])
      c. Call the LLM:
             system = "You are an expert executor. Implement the plan thoroughly. Be specific and complete."
             messages = [{"role": "user", "content":
                 f"Plan:\\n{state['plan']}\\n\\nResearch context:\\n{context}\\n\\nExecute the plan."}]
             result = llm_call(messages, system=system, max_tokens=1500)
      d. Print: print(f"  ✅ Execution complete ({len(result.split())} words)")
      e. Return: {"execution_result": result, "steps": state["steps"] + 1}
    """
    # TODO 4: implement execute node
    raise NotImplementedError("execute_node() not implemented yet")


def reject_node(state: ResearchState) -> dict:
    """Handle plan rejection (already implemented)."""
    return {
        "execution_result": "Plan was rejected by the operator. No action was taken.",
        "steps": state["steps"] + 1,
    }


# ── Routing Functions ──────────────────────────────────────────────────────────

def route_after_plan(state: ResearchState) -> Literal["human_approval", "execute"]:
    """
    Route to human approval for MEDIUM/HIGH risk; auto-execute for LOW risk.

    TODO 3:
      if state["risk_level"] in {"MEDIUM", "HIGH"}:
          return "human_approval"
      return "execute"
    """
    # TODO 3: implement routing function
    raise NotImplementedError("route_after_plan() not implemented yet")


def route_after_approval(state: ResearchState) -> Literal["execute", "reject"]:
    """Route based on human decision (already implemented)."""
    return "execute" if state.get("approved") else "reject"


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_agent():
    """
    Assemble the StateGraph and compile it with MemorySaver checkpointing.

    TODO 5 — build the complete graph:

      a. Create the graph:
             workflow = StateGraph(ResearchState)

      b. Add all nodes:
             workflow.add_node("research",       research_node)
             workflow.add_node("plan",            plan_node)
             workflow.add_node("human_approval",  human_approval_node)
             workflow.add_node("execute",         execute_node)
             workflow.add_node("reject",          reject_node)

      c. Set entry point:
             workflow.set_entry_point("research")

      d. Add unconditional edges (always go from A to B):
             workflow.add_edge("research", "plan")

      e. Add conditional edge from "plan" (uses route_after_plan):
             workflow.add_conditional_edges(
                 "plan",
                 route_after_plan,
                 {"human_approval": "human_approval", "execute": "execute"},
             )

      f. Add conditional edge from "human_approval" (uses route_after_approval):
             workflow.add_conditional_edges(
                 "human_approval",
                 route_after_approval,
                 {"execute": "execute", "reject": "reject"},
             )

      g. Add terminal edges to END (import END from langgraph.graph):
             workflow.add_edge("execute", END)
             workflow.add_edge("reject",  END)

      h. Compile with MemorySaver:
             memory = MemorySaver()
             return workflow.compile(checkpointer=memory)

    With MemorySaver you can resume an interrupted run by providing the same
    thread_id in the config dict:
        config = {"configurable": {"thread_id": "run-001"}}
        agent.invoke(initial_state, config=config)
    """
    if not HAS_LANGGRAPH:
        raise ImportError("Install langgraph: pip install langgraph")

    # TODO 5: build and compile the graph
    raise NotImplementedError("build_agent() not implemented yet")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(query: str, thread_id: str = "default"):
    agent = build_agent()

    initial_state: ResearchState = {
        "query":            query,
        "research_results": [],
        "plan":             "",
        "risk_level":       "LOW",
        "approved":         False,
        "execution_result": "",
        "steps":            0,
    }

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n🤖 LangGraph Decision Agent")
    print(f"🔎 Query: {query}\n")

    final = agent.invoke(initial_state, config=config)

    print(f"\n{'='*55}")
    print("📄 FINAL RESULT:")
    print(f"{'='*55}")
    print(final["execution_result"])
    print(f"\nSteps taken: {final['steps']}")
    return final


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if sys.argv[1:] else "Summarise the top 3 multi-agent design patterns"
    run(query)
