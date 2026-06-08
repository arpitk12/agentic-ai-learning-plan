"""Project 19 main entry points — CLI and FastAPI server."""
from __future__ import annotations

import argparse
import asyncio
import logging

from src.config import cfg

logging.basicConfig(level=cfg.log_level)


# ── CLI: interactive review ───────────────────────────────────────────────

def run_cli(code: str, language: str = "python", thread_id: str = "demo") -> None:
    """Run a code review interactively in the terminal."""
    from langgraph.types import Command
    from src.graph.checkpointer import get_checkpointer
    from src.graph.graph import build_graph

    config = {"configurable": {"thread_id": thread_id}}

    with get_checkpointer() as cp:
        graph = build_graph(checkpointer=cp)

        print("\n🔍 Starting code review...\n")
        result = asyncio.run(
            graph.ainvoke({"code": code, "language": language, "messages": []}, config=config)
        )

        print("=" * 60)
        print(f"Security Issues: {result.get('security_issues', [])}")
        print(f"Quality Score:   {result.get('quality_score', 0):.1f}/10")
        print("\n📋 REVIEW:\n")
        print(result.get("review", "No review generated"))
        print("=" * 60)

        # Human-in-the-loop
        while result.get("status") == "review_ready":
            fb = input("\n> Approve ('approve') or request changes ('revise: <notes>'): ").strip()
            result = asyncio.run(graph.ainvoke(Command(resume=fb), config=config))
            if result.get("status") == "needs_revision":
                print("\n🔄 Revising review...\n")
                print(result.get("review", ""))
            else:
                print("\n✅ Review approved and finalized!")
                break


# ── FastAPI server ─────────────────────────────────────────────────────────

def run_api() -> None:
    import uvicorn
    from fastapi import FastAPI
    from src.api.routes import router

    app = FastAPI(title="LangGraph Code Review API", version="1.0.0")
    app.include_router(router)

    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Code Review")
    sub = parser.add_subparsers(dest="cmd")

    cli_p = sub.add_parser("review", help="Review a Python file")
    cli_p.add_argument("file", help="Path to the Python file to review")
    cli_p.add_argument("--language", default="python")
    cli_p.add_argument("--thread-id", default="demo")

    sub.add_parser("api", help="Start FastAPI server")

    args = parser.parse_args()
    if args.cmd == "review":
        with open(args.file) as f:
            code = f.read()
        run_cli(code, args.language, args.thread_id)
    elif args.cmd == "api":
        run_api()
    else:
        parser.print_help()
