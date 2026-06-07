"""
SOLUTION — Project 10: LangGraph Decision Agent (HITL + Checkpointing)

StateGraph flow:
  research → plan → [route by risk] → human_approval? → execute / reject

Run:
    pip install langgraph
    python solution.py "Summarise the top 3 multi-agent design patterns"
    python solution.py "Draft a public announcement about AI safety"
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
from typing import TypedDict, Annotated, Literal
import operator
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ── State Schema ───────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:            str
    research_results: Annotated[list[str], operator.add]
    plan:             str
    risk_level:       str
    approved:         bool
    execution_result: str
    steps:            int


# ── Helpers ────────────────────────────────────────────────────────────────────

def llm_call(messages: list[dict], system: str = "", max_tokens: int = 1024) -> str:
    r = chat(messages, system=system, max_tokens=max_tokens)
    return get_text(r)


def web_search(query: str) -> str:
    snippets = {
        "multi-agent": (
            "Key multi-agent patterns: (1) Orchestrator-Worker — planner delegates to specialists. "
            "(2) Debate/Adversarial — agents argue pro/con, judge decides. "
            "(3) Fan-Out/Fan-In — parallel processing with asyncio.gather(). "
            "(4) Map-Reduce — large dataset processed in parallel batches."
        ),
        "vector database": (
            "Vector DB trade-offs: FAISS (fast, in-memory, no filtering), "
            "ChromaDB (dev-friendly, <5M vectors), "
            "Qdrant (production-grade, advanced filtering, billions of vectors), "
            "pgvector (SQL+vector, ACID transactions)."
        ),
        "langgraph": (
            "LangGraph StateGraph: TypedDict state, nodes are functions, edges are transitions. "
            "MemorySaver checkpointing allows resumption. interrupt_before enables HITL pauses."
        ),
    }
    for key, result in snippets.items():
        if key.lower() in query.lower():
            return result
    return (
        f"Research on '{query}': This topic involves important trade-offs, "
        "best practices, and production considerations that practitioners should understand."
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

def research_node(state: ResearchState) -> dict:
    print(f"🔍 Researching: {state['query'][:60]}...")
    result = web_search(state["query"])
    return {
        "research_results": [result],
        "steps": state["steps"] + 1,
    }


def plan_node(state: ResearchState) -> dict:
    print("📋 Creating plan...")
    context = "\n\n".join(state["research_results"])
    system = (
        "You are a strategic planner. Create a concise action plan based on the research. "
        "Classify risk: LOW (analysis/summary only), MEDIUM (creates content, no external action), "
        "HIGH (sends emails, posts publicly, modifies data). "
        'Return ONLY valid JSON: {"plan": "...", "risk_level": "LOW|MEDIUM|HIGH", "reasoning": "..."}'
    )
    messages = [{"role": "user", "content":
        f"Query: {state['query']}\n\nResearch:\n{context}\n\nCreate a plan."}]
    raw = llm_call(messages, system=system, max_tokens=600)

    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        data = json.loads(raw[s:e])
    except (json.JSONDecodeError, ValueError):
        data = {
            "plan": f"Analyse and summarise key points about: {state['query']}",
            "risk_level": "LOW",
            "reasoning": "Defaulting to LOW risk (JSON parse failed)",
        }

    print(f"  Risk: {data['risk_level']} — {data.get('reasoning', '')[:80]}")
    return {
        "plan":       data["plan"],
        "risk_level": data["risk_level"],
        "steps":      state["steps"] + 1,
    }


def human_approval_node(state: ResearchState) -> dict:
    print(f"\n{'='*55}")
    print("⚠️  HUMAN APPROVAL REQUIRED")
    print(f"{'='*55}")
    print(f"Risk level: {state['risk_level'].upper()}")
    print(f"\nProposed plan:\n{state['plan']}")
    print(f"{'='*55}")
    answer = input("Approve this plan? [y/n]: ").strip().lower()
    approved = answer == "y"
    print("✅ Approved — proceeding." if approved else "❌ Rejected by operator.")
    return {"approved": approved, "steps": state["steps"] + 1}


def execute_node(state: ResearchState) -> dict:
    print("⚙️  Executing plan...")
    context = "\n\n".join(state["research_results"][-2:])
    system = "You are an expert executor. Implement the plan thoroughly. Be specific and complete."
    messages = [{"role": "user", "content":
        f"Plan:\n{state['plan']}\n\nResearch context:\n{context}\n\nExecute the plan."}]
    result = llm_call(messages, system=system, max_tokens=1500)
    print(f"  ✅ Execution complete ({len(result.split())} words)")
    return {"execution_result": result, "steps": state["steps"] + 1}


def reject_node(state: ResearchState) -> dict:
    return {
        "execution_result": "Plan was rejected by the operator. No action was taken.",
        "steps": state["steps"] + 1,
    }


# ── Routing Functions ──────────────────────────────────────────────────────────

def route_after_plan(state: ResearchState) -> Literal["human_approval", "execute"]:
    if state["risk_level"] in {"MEDIUM", "HIGH"}:
        return "human_approval"
    return "execute"


def route_after_approval(state: ResearchState) -> Literal["execute", "reject"]:
    return "execute" if state.get("approved") else "reject"


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_agent():
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("research",      research_node)
    workflow.add_node("plan",          plan_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute",       execute_node)
    workflow.add_node("reject",        reject_node)

    # Entry point
    workflow.set_entry_point("research")

    # Unconditional edges
    workflow.add_edge("research", "plan")

    # Conditional: plan → approve (HIGH/MED) or execute (LOW)
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {"human_approval": "human_approval", "execute": "execute"},
    )

    # Conditional: approval → execute or reject
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"execute": "execute", "reject": "reject"},
    )

    # Terminal edges
    workflow.add_edge("execute", END)
    workflow.add_edge("reject",  END)

    # Compile with checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(query: str, thread_id: str = "default"):
    agent = build_agent()

    # Print graph structure
    try:
        print("\n📊 Graph structure:")
        print(agent.get_graph().draw_ascii())
    except Exception:
        pass

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
