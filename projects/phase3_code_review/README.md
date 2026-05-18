# Project 3 — Multi-Agent Code Review System

## Brief
A system where a user submits a GitHub PR URL. An orchestrator spawns
4 parallel subagents to review different aspects. Results are aggregated
into a final scored review report.

## Requirements
- [ ] Accept GitHub PR URL as input
- [ ] Fetch PR diff via GitHub API
- [ ] 4 parallel subagents (asyncio.gather):
  - Security Reviewer — looks for injection, hardcoded secrets, etc.
  - Performance Reviewer — O(n) issues, unnecessary DB calls, etc.
  - Style Checker — naming conventions, docstrings, complexity
  - Test Coverage Analyst — missing test cases, untested branches
- [ ] Orchestrator aggregates all reviews into final report
- [ ] Overall score (0-100) with breakdown by category
- [ ] Output as JSON + human-readable markdown

## Setup
```bash
pip install anthropic httpx python-dotenv pydantic asyncio
```

Set in `.env`:
```
ANTHROPIC_API_KEY=your_key
GITHUB_TOKEN=your_personal_access_token
```

## Usage
```bash
python starter.py https://github.com/owner/repo/pull/123
# Outputs: review_PR123.json and review_PR123.md
```

## Architecture
```
Input: PR URL
    ↓
[PR Fetcher] → diff, files changed, PR description
    ↓
[Orchestrator] ─────────────────────────────────────
    ├─ [Security Agent]   ─── async ──→ security_findings
    ├─ [Performance Agent]─── async ──→ perf_findings
    ├─ [Style Agent]      ─── async ──→ style_findings
    └─ [Test Agent]       ─── async ──→ test_findings
    ↓ (all gathered)
[Aggregator] → final_report (score + summary + per-category)
```

## Hints
- Use `asyncio.gather(*[agent(diff) for agent in agents])`
- Each subagent gets the same diff but a different system prompt
- Aggregator is another LLM call that synthesizes 4 reviews
- Score: Security 40pts, Performance 25pts, Style 20pts, Tests 15pts
