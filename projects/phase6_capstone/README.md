# Project 6 — Capstone: Full-Stack Agentic SaaS

## Choose Your Product
Pick ONE of these or propose your own:

**Option A — AI Customer Support Agent**
Handles support tickets, looks up order history, drafts responses,
escalates to human when confidence is low.

**Option B — Autonomous Data Analyst**
Upload a CSV, ask questions in plain English, agent writes and runs
Python analysis code, returns charts + insights.

**Option C — Multi-Agent Content Pipeline**
Input: topic + brand voice doc. Output: blog post + social posts + email.
3 specialized agents: Researcher → Writer → Editor.

---

## Non-Negotiable Requirements (all options)
- [ ] REST API (FastAPI) with background job queue
- [ ] RAG over a knowledge base (product docs, data, brand guide)
- [ ] At least 2 specialized subagents
- [ ] Per-user cost tracking + budget limits
- [ ] Eval suite (min 20 test cases, run in CI)
- [ ] Structured logging (structlog or OpenTelemetry)
- [ ] Docker Compose deployment
- [ ] Grafana dashboard: cost / latency / error rate / queue depth
- [ ] README with architecture diagram and setup guide

## Suggested Stack
```
Backend:    FastAPI + Celery + Redis
Agent:      LangGraph + Anthropic SDK
RAG:        Chroma + sentence-transformers
Database:   PostgreSQL (runs + memory) + Redis (jobs + cache)
Monitoring: Prometheus + Grafana
Deploy:     Docker Compose (local) or Railway/Render (cloud)
```

## Deliverables
1. `README.md` — architecture diagram, setup guide, design decisions
2. `docker-compose.yml` — runs the whole system with one command
3. `tests/eval/` — 20+ eval cases, passes in CI
4. `grafana/` — dashboard JSON export
5. Working demo (loom video or live URL)

## Evaluation Rubric
| Area | Weight |
|---|---|
| Core agent functionality works | 30% |
| Production infrastructure (queue, streaming, Docker) | 25% |
| Eval suite + CI passes | 20% |
| Observability (logs, metrics, tracing) | 15% |
| Code quality + documentation | 10% |

## Timeline
- Day 1-2: Architecture design, scaffold, Docker Compose
- Day 3-5: Core agent logic + RAG
- Day 6-7: API endpoints + streaming
- Day 8-9: Evals + CI
- Day 10: Monitoring + polish + demo
