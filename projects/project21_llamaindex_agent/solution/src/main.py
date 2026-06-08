"""Main entry point for project 21 — LlamaIndex Multi-Doc RAG Agent."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex Multi-Doc RAG Agent")
    sub = parser.add_subparsers(dest="cmd")

    ingest_p = sub.add_parser("ingest", help="Ingest documents into vector index")
    ingest_p.add_argument("path", help="Path to directory containing documents")

    query_p = sub.add_parser("query", help="Query the knowledge base")
    query_p.add_argument("question", help="Question to ask")
    query_p.add_argument("--mode", choices=["vector", "summary", "router", "subq", "agent"],
                         default="agent")

    sub.add_parser("chat", help="Start interactive REPL with the ReAct agent")

    args = parser.parse_args()

    if args.cmd == "ingest":
        from src.indexing.pipeline import load_or_build_index
        index = load_or_build_index(args.path)
        print(f"✅ Index ready with {len(index.docstore.docs)} nodes")

    elif args.cmd == "query":
        from src.indexing.pipeline import configure_settings, load_or_build_index
        configure_settings()
        index = load_or_build_index("data/docs")

        if args.mode == "vector":
            engine = index.as_query_engine(similarity_top_k=5)
        elif args.mode == "router":
            from llama_index.core import SummaryIndex
            summary_idx = SummaryIndex(list(index.docstore.docs.values()))
            from src.query.engines import build_router_engine
            engine = build_router_engine(index, summary_idx)
        elif args.mode == "agent":
            from llama_index.core.tools import QueryEngineTool
            from src.agent.agent import build_react_agent, make_calculator_tool
            tool = QueryEngineTool.from_defaults(
                query_engine=index.as_query_engine(similarity_top_k=5),
                description="Answers questions from the document knowledge base.",
            )
            engine = build_react_agent([tool], extra_tools=[make_calculator_tool()])
        else:
            engine = index.as_query_engine()

        response = engine.query(args.question)
        print(f"\n💬 Answer: {response}\n")

    elif args.cmd == "chat":
        from src.indexing.pipeline import configure_settings, load_or_build_index
        from llama_index.core.tools import QueryEngineTool
        from src.agent.agent import build_react_agent, make_calculator_tool

        configure_settings()
        index = load_or_build_index("data/docs")
        tool = QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(similarity_top_k=5),
            description="Knowledge base of loaded documents.",
        )
        agent = build_react_agent([tool], extra_tools=[make_calculator_tool()])

        print("🤖 LlamaIndex Agent ready. Type 'quit' to exit.\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            response = agent.chat(q)
            print(f"Agent: {response}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
