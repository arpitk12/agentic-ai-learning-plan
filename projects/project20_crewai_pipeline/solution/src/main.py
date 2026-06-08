"""Main entry point for project 20 — CrewAI Content Pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI Content Pipeline")
    parser.add_argument("topic", help="Topic to research and write about")
    parser.add_argument(
        "--mode",
        choices=["sequential", "hierarchical", "async"],
        default="sequential",
        help="Pipeline execution mode",
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting CrewAI pipeline ({args.mode}) for: {args.topic}\n")

    if args.mode == "sequential":
        from src.crew.crew import run_sequential_pipeline
        result = run_sequential_pipeline(args.topic)
    elif args.mode == "hierarchical":
        from src.crew.crew_hierarchical import run_hierarchical_pipeline
        result = run_hierarchical_pipeline(args.topic)
    else:
        import asyncio
        from src.crew.crew import run_async_pipeline
        result = asyncio.run(run_async_pipeline(args.topic))

    # Save outputs
    out_path = Path(args.output) / "result.json"
    with open(out_path, "w") as f:
        json.dump({k: str(v) for k, v in result.items()}, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # Print SEO report if available
    if result.get("seo"):
        seo = result["seo"]
        print(f"\n📊 SEO Score: {seo.seo_score}/100")
        print(f"   Keyword: {seo.primary_keyword}")
        print(f"   Meta title: {seo.meta_title}")


if __name__ == "__main__":
    main()
