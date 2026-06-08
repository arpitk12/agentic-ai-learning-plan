"""Main entry point — CLI and API for the compliance review agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path


def run_review(doc_id: str, doc_content: str, doc_type: str = "contract") -> dict:
    """Run a full compliance review and return the final state."""
    from langgraph.types import Command
    from src.graph.graph import get_graph
    from src.observability.tracer import configure_langsmith

    configure_langsmith()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {
        "document_id": doc_id,
        "document_content": doc_content,
        "document_type": doc_type,
        "messages": [],
        "audit_entries": [],
        "llm_cost_usd": 0.0,
        "processing_time_seconds": 0.0,
        "compliance_score": 0.0,
        "risk_assessment": {},
        "policy_check": {},
        "human_feedback": None,
        "human_reviewer": None,
        "escalated_to_legal": False,
        "status": "pending",
    }

    print(f"\n🔍 Starting compliance review for {doc_id}...\n")
    with get_graph() as graph:
        result = asyncio.run(graph.ainvoke(state_input, config=config))

        # If graph paused for HITL
        while result.get("status") == "pending_review":
            risk = result.get("risk_assessment", {})
            print(f"\n⚠️  HUMAN REVIEW REQUIRED")
            print(f"   Risk level:    {risk.get('risk_level', 'unknown').upper()}")
            print(f"   Risk factors:  {', '.join(risk.get('risk_factors', []))}")
            print(f"   Score:         {result.get('compliance_score', 0):.0f}/100")
            fb = input("\n> Respond (approve: <notes> | reject: <notes> | escalate: <notes>): ").strip()
            result = asyncio.run(graph.ainvoke(Command(resume=fb), config=config))

        return result


def run_api() -> None:
    """Start FastAPI service for the compliance agent."""
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticBase

    app = FastAPI(title="Compliance Review API", version="1.0.0")

    class ReviewRequest(PydanticBase):
        document_id: str
        document_content: str
        document_type: str = "contract"

    @app.post("/review")
    async def review(req: ReviewRequest) -> dict:
        from src.graph.graph import get_graph
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "document_id": req.document_id, "document_content": req.document_content,
            "document_type": req.document_type, "messages": [], "audit_entries": [],
            "llm_cost_usd": 0.0, "processing_time_seconds": 0.0,
            "compliance_score": 0.0, "risk_assessment": {}, "policy_check": {},
            "human_feedback": None, "human_reviewer": None,
            "escalated_to_legal": False, "status": "pending",
        }
        with get_graph() as graph:
            result = await graph.ainvoke(state, config=config)
        return {
            "thread_id": thread_id,
            "status": result.get("status"),
            "risk_level": result.get("risk_assessment", {}).get("risk_level"),
            "compliance_score": result.get("compliance_score"),
            "llm_cost_usd": result.get("llm_cost_usd"),
        }

    @app.get("/report")
    def cost_report() -> dict:
        from src.reporting.cost_report import CostComparison
        return CostComparison().to_dict()

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise Compliance Review Agent")
    sub = parser.add_subparsers(dest="cmd")

    rev_p = sub.add_parser("review", help="Review a document")
    rev_p.add_argument("--doc-id", required=True)
    rev_p.add_argument("--doc-file", help="Path to document text file")
    rev_p.add_argument("--doc-content", help="Document content as string")
    rev_p.add_argument("--type", default="contract", dest="doc_type")
    rev_p.add_argument("--output", default="output/review_result.json")

    sub.add_parser("api", help="Start FastAPI server")
    sub.add_parser("report", help="Print Q1 cost savings report")

    args = parser.parse_args()

    if args.cmd == "review":
        content = args.doc_content or (Path(args.doc_file).read_text() if args.doc_file else "")
        result = run_review(args.doc_id, content, args.doc_type)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
        print(f"\n✅ Status: {result.get('status')} | Cost: ${result.get('llm_cost_usd', 0):.4f}")
        print(f"   Result saved to {args.output}")
    elif args.cmd == "api":
        run_api()
    elif args.cmd == "report":
        from src.reporting.cost_report import CostComparison
        CostComparison().print_report()
    else:
        parser.print_help()
