# Project 33 — Multi-Tenant Agent Platform

> **Stack**: LangGraph · FastAPI · Redis · JWT · Langfuse  
> **Phase 7 — Advanced Production** | Priority: P2 🟡

---

## What You'll Build

A fully isolated multi-tenant agent platform where multiple organizations ("tenants") share the same deployment but have complete data isolation, independent rate limits, per-tenant cost tracking, and role-based capability access.

---

## Why Multi-Tenancy Is Hard for Agents

| Problem | Consequence |
|---|---|
| Shared LangGraph checkpointer namespace | Tenant A can read Tenant B's conversation history |
| Shared vector store collection | Tenant A's documents appear in Tenant B's RAG results |
| No per-tenant rate limits | One tenant can monopolize the service |
| Shared cost tracking | Can't bill tenants accurately or set budgets |
| Same capabilities for all tenants | Free tier users can access premium features |

---

## Milestones

### Milestone 1 — Tenant Model + JWT Auth
Define `Tenant` (id, name, tier, rate_limit, allowed_capabilities). Issue JWT tokens scoped to tenant. Middleware extracts tenant from every request.

### Milestone 2 — State Isolation in LangGraph
Namespace every `thread_id` and `checkpoint_ns` with `tenant_id`. Verify: Tenant A's graph state cannot be accessed by Tenant B even with direct API calls.

### Milestone 3 — Per-Tenant RAG Isolation
Create tenant-specific ChromaDB collections (or Qdrant namespaces). Ingest different documents per tenant. Verify cross-tenant retrieval is impossible.

### Milestone 4 — Rate Limiting
Implement per-tenant rate limiting using Redis (token bucket algorithm): free=10 req/min, pro=100 req/min, enterprise=1000 req/min. Return HTTP 429 with Retry-After header.

### Milestone 5 — Capability RBAC
Define capabilities: `[basic_review, advanced_review, bulk_processing, audit_export, custom_policies]`. Map to tiers. Decorator `@require_capability("advanced_review")` on route handlers.

### Milestone 6 — Per-Tenant Cost Tracking
Tag every Langfuse trace with `tenant_id`. Build a monthly cost report per tenant. Add budget alerts: warn at 80%, block at 100%.

### Milestone 7 — Admin Dashboard API
Build admin endpoints (require `admin` JWT claim):
- `GET /admin/tenants` — list all tenants + usage stats
- `GET /admin/tenants/{id}/usage` — monthly cost + request count
- `POST /admin/tenants/{id}/budget` — set monthly budget
- `PUT /admin/tenants/{id}/tier` — upgrade/downgrade tier

---

## Setup

```bash
pip install langgraph fastapi uvicorn pyjwt redis chromadb langfuse litellm pydantic python-dotenv
docker run -p 6379:6379 redis:7
```

---

## Expected Output

```
=== Multi-Tenant Isolation Test ===

Tenant A (Acme Corp, pro tier): DOC-A1, DOC-A2 loaded
Tenant B (Beta LLC, free tier):  DOC-B1 loaded

Test 1: Tenant A RAG search
  Query: "vendor agreements"
  Results: [DOC-A1, DOC-A2] ✅ (only A's documents)

Test 2: Tenant B RAG search
  Query: "vendor agreements"
  Results: [DOC-B1] ✅ (only B's documents, A's not visible)

Test 3: Rate limit enforcement (free tier)
  Requests 1-10: 200 OK
  Request 11:    429 Too Many Requests (Retry-After: 45s) ✅

Test 4: Capability enforcement
  Tenant B (free) → POST /bulk-process: 403 Forbidden ✅
  Tenant A (pro)  → POST /bulk-process: 200 OK ✅

Cost tracking (30 days):
  Acme Corp (pro):  $127.40 of $500 budget (25.5%)
  Beta LLC (free):  $8.20 of $20 budget   (41.0%) ⚠️ alert at 80%
```
