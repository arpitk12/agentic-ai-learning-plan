"""
Project 3 Starter — Multi-Agent Code Review System
Fill in the TODOs to complete the project.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import asyncio
import json
import re
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


# --- Models ---
class AgentReview(BaseModel):
    agent: str
    findings: list[str]
    score: int          # 0-100
    critical_issues: list[str]


class FinalReport(BaseModel):
    pr_url: str
    overall_score: int
    summary: str
    reviews: list[AgentReview]
    top_issues: list[str]


# --- GitHub PR Fetcher ---
def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse 'https://github.com/owner/repo/pull/123' → (owner, repo, 123)"""
    # TODO: use regex to extract owner, repo, pr_number
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not match:
        raise ValueError(f"Invalid PR URL: {url}")
    return match.group(1), match.group(2), int(match.group(3))


async def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """TODO: Fetch PR diff from GitHub API."""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff"
    }
    async with httpx.AsyncClient() as http:
        # TODO: GET https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}
        # for diff: use Accept: application/vnd.github.v3.diff
        pass
    return "diff not implemented"


# --- Subagent System Prompts ---
AGENT_PROMPTS = {
    "security": """You are a security code reviewer. Analyze the diff for:
- SQL/command injection vulnerabilities
- Hardcoded secrets, API keys, passwords
- Insecure deserialization
- Missing authentication/authorization checks
- Sensitive data exposure
Score 0-100 (100 = no security issues). List specific findings with line references.""",

    "performance": """You are a performance code reviewer. Analyze the diff for:
- O(n²) or worse algorithms where O(n) is possible
- N+1 database query patterns
- Missing indexes or inefficient queries
- Unnecessary data loading (fetch all when you need one)
- Memory leaks or excessive object creation
Score 0-100 (100 = no performance issues).""",

    "style": """You are a code style reviewer. Analyze the diff for:
- Naming convention violations
- Missing docstrings on public functions/classes
- Functions over 50 lines (should be split)
- Magic numbers (use named constants)
- Code duplication (DRY violations)
Score 0-100 (100 = perfect style).""",

    "tests": """You are a test coverage reviewer. Analyze the diff for:
- New functions without corresponding tests
- Missing edge case coverage (empty input, None, boundary values)
- Untested error paths
- Missing integration tests for new APIs
- Test quality (are assertions meaningful?)
Score 0-100 (100 = excellent test coverage).""",
}


# --- Subagent Runner ---
async def run_subagent(agent_name: str, diff: str) -> AgentReview:
    """TODO: Run one subagent against the diff. Return AgentReview."""
    system = AGENT_PROMPTS[agent_name]
    prompt = f"""Review this PR diff:

```diff
{diff[:8000]}  
```

Respond with JSON only:
{{
  "findings": ["finding1", "finding2"],
  "score": <0-100>,
  "critical_issues": ["critical1"]
}}"""

    # TODO: call client.messages.create (async)
    # TODO: parse JSON response
    # TODO: return AgentReview

    return AgentReview(
        agent=agent_name,
        findings=["Not implemented"],
        score=50,
        critical_issues=[]
    )


# --- Orchestrator ---
async def orchestrate(pr_url: str) -> FinalReport:
    """TODO: Fetch PR, run 4 agents in parallel, aggregate."""
    print(f"Fetching PR: {pr_url}")
    owner, repo, pr_num = parse_pr_url(pr_url)
    diff = await fetch_pr_diff(owner, repo, pr_num)
    print(f"Diff length: {len(diff)} chars")

    print("Running 4 review agents in parallel...")
    # TODO: use asyncio.gather to run all 4 agents simultaneously
    agent_names = ["security", "performance", "style", "tests"]
    reviews = []
    for name in agent_names:
        reviews.append(await run_subagent(name, diff))

    # TODO: Calculate weighted overall score
    # security=40%, performance=25%, style=20%, tests=15%
    weights = {"security": 0.40, "performance": 0.25, "style": 0.20, "tests": 0.15}
    overall = sum(r.score * weights[r.agent] for r in reviews)

    # TODO: LLM call to generate summary from all reviews

    return FinalReport(
        pr_url=pr_url,
        overall_score=int(overall),
        summary="Not implemented",
        reviews=reviews,
        top_issues=[]
    )


# --- Main ---
async def main():
    pr_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/anthropics/anthropic-sdk-python/pull/1"
    report = await orchestrate(pr_url)

    # Save outputs
    pr_id = pr_url.split("/")[-1]
    json_path = f"review_PR{pr_id}.json"
    with open(json_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"\n{'='*50}")
    print(f"Overall Score: {report.overall_score}/100")
    print(f"Summary: {report.summary}")
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
