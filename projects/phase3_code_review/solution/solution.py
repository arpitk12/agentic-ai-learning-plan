"""
SOLUTION — Project 3: Multi-Agent Code Review System
4 parallel specialist reviewers + orchestrator aggregation.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import asyncio
import json
import re
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()


# ── Data Models ────────────────────────────────────────────────────────────────

class ReviewCategory(BaseModel):
    category: str
    score: int           # 0-25 each (total = 100)
    issues: list[str]
    suggestions: list[str]
    severity: str        # "low" | "medium" | "high" | "critical"


class CodeReviewReport(BaseModel):
    pr_url: str
    overall_score: int   # 0-100
    summary: str
    security: ReviewCategory
    performance: ReviewCategory
    style: ReviewCategory
    testing: ReviewCategory


# ── GitHub PR Fetcher ──────────────────────────────────────────────────────────

def fetch_pr_diff(pr_url: str) -> dict:
    """Fetch PR diff from GitHub API."""
    # Parse: https://github.com/owner/repo/pull/123
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        raise ValueError(f"Invalid PR URL: {pr_url}")
    owner, repo, pr_num = match.groups()

    import httpx
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3.diff"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=15) as http:
            # Get PR metadata
            meta_r = http.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}",
                headers={**headers, "Accept": "application/vnd.github.v3+json"},
            )
            meta = meta_r.json()

            # Get diff
            diff_r = http.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}",
                headers=headers,
            )
            return {
                "title": meta.get("title", ""),
                "description": meta.get("body", "") or "",
                "diff": diff_r.text[:8000],  # cap at 8K chars
                "files_changed": meta.get("changed_files", 0),
                "additions": meta.get("additions", 0),
                "deletions": meta.get("deletions", 0),
            }
    except Exception as e:
        # Return mock diff for testing
        return {
            "title": "Test PR",
            "description": "Adding new feature",
            "diff": """
+def process_user_data(user_id, db_conn):
+    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection!
+    result = db_conn.execute(query)
+    password = "hardcoded_secret_123"  # hardcoded credential
+    data = []
+    for row in result:
+        for item in db_conn.execute("SELECT * FROM orders"):  # N+1 query
+            data.append((row, item))
+    return data
""",
            "files_changed": 1,
            "additions": 8,
            "deletions": 0,
        }


# ── Specialist Agents ──────────────────────────────────────────────────────────

async def _review_agent(diff: str, meta: dict, role: str, focus: str, max_score: int = 25) -> ReviewCategory:
    system = f"""You are a {role}. Analyze the code diff for {focus}.
Return ONLY valid JSON (no markdown):
{{
  "score": <0-{max_score}>,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"],
  "severity": "low|medium|high|critical"
}}
Be specific and actionable."""

    prompt = (
        f"PR: {meta['title']}\n"
        f"Description: {meta['description'][:300]}\n\n"
        f"Diff:\n{diff}"
    )

    r = await achat([{"role": "user", "content": prompt}], system=system, max_tokens=1024)
    raw = get_text(r)
    s = raw.find("{"); e = raw.rfind("}") + 1
    data = json.loads(raw[s:e])
    return ReviewCategory(category=role, **data)


async def security_review(diff: str, meta: dict) -> ReviewCategory:
    return await _review_agent(diff, meta,
        "Senior Security Engineer",
        "SQL injection, XSS, hardcoded secrets, insecure deserialization, auth issues, OWASP Top 10")


async def performance_review(diff: str, meta: dict) -> ReviewCategory:
    return await _review_agent(diff, meta,
        "Performance Engineer",
        "N+1 queries, O(n²) algorithms, memory leaks, blocking I/O, inefficient data structures")


async def style_review(diff: str, meta: dict) -> ReviewCategory:
    return await _review_agent(diff, meta,
        "Senior Software Engineer",
        "naming conventions, docstrings, function length, complexity, code duplication, readability")


async def testing_review(diff: str, meta: dict) -> ReviewCategory:
    return await _review_agent(diff, meta,
        "QA Engineer",
        "missing unit tests, untested edge cases, no error handling tests, missing mocks")


# ── Orchestrator ───────────────────────────────────────────────────────────────

async def review_pr(pr_url: str) -> CodeReviewReport:
    print(f"📥 Fetching PR: {pr_url}")
    meta = fetch_pr_diff(pr_url)
    diff = meta["diff"]
    print(f"   Files: {meta['files_changed']} | +{meta['additions']} -{meta['deletions']}")

    print("🔍 Running 4 parallel reviews...")
    security, performance, style, testing = await asyncio.gather(
        security_review(diff, meta),
        performance_review(diff, meta),
        style_review(diff, meta),
        testing_review(diff, meta),
    )

    overall = security.score + performance.score + style.score + testing.score

    # Synthesize summary
    all_issues = security.issues + performance.issues + style.issues + testing.issues
    r = await achat(
        [{"role": "user", "content":
            f"Write a 2-sentence executive summary of this code review (score {overall}/100). "
            f"Issues found: {json.dumps(all_issues[:8])}"}],
        max_tokens=300,
    )
    summary = get_text(r)

    return CodeReviewReport(
        pr_url=pr_url,
        overall_score=overall,
        summary=summary,
        security=security,
        performance=performance,
        style=style,
        testing=testing,
    )


def print_report(report: CodeReviewReport):
    print(f"\n{'='*60}")
    print(f"CODE REVIEW REPORT — Score: {report.overall_score}/100")
    print(f"{'='*60}")
    print(f"\nSummary: {report.summary}")
    for cat in [report.security, report.performance, report.style, report.testing]:
        print(f"\n{'─'*40}")
        print(f"[{cat.category}] Score: {cat.score}/25 | Severity: {cat.severity.upper()}")
        for issue in cat.issues:
            print(f"  ⚠ {issue}")
        for sug in cat.suggestions:
            print(f"  💡 {sug}")


async def main():
    pr_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/example/repo/pull/1"
    report = await review_pr(pr_url)
    print_report(report)

    # Save to files
    slug = re.sub(r"[^\w]", "_", pr_url.split("/")[-1])
    Path(f"review_{slug}.json").write_text(report.model_dump_json(indent=2))
    print(f"\n✅ Saved to review_{slug}.json")


if __name__ == "__main__":
    asyncio.run(main())
