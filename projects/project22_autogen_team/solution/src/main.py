"""Main entry point for project 22 — AutoGen Coding Team."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen Multi-Agent Coding Team")
    sub = parser.add_subparsers(dest="cmd")

    two_p = sub.add_parser("simple", help="Two-agent simple coding task")
    two_p.add_argument("task", help="Coding task description")

    team_p = sub.add_parser("team", help="Run full 5-agent team")
    team_p.add_argument("task", help="Feature or project to build")
    team_p.add_argument("--max-rounds", type=int, default=20)
    team_p.add_argument("--work-dir", default="workspace")
    team_p.add_argument("--output", default="output/team_result.json")

    args = parser.parse_args()

    if args.cmd == "simple":
        from src.team.nested_chat import run_two_agent_chat
        result = run_two_agent_chat(args.task)
        print(f"\n✅ Result:\n{result}")

    elif args.cmd == "team":
        from src.team.groupchat import run_team
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        Path("output").mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Starting AutoGen team for:\n{args.task}\n")
        result = run_team(args.task, max_rounds=args.max_rounds, work_dir=args.work_dir)

        print(f"\n✅ Status: {result['final_status']} ({result['rounds_used']} rounds)")
        with open(args.output, "w") as f:
            json.dump(result["chat_history"], f, indent=2, default=str)
        print(f"   Chat history saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
