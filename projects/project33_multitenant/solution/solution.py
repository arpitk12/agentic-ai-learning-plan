"""
Project 33 SOLUTION — Multi-Tenant Compliance Agent
Per-tenant isolation: namespace + RBAC + Redis rate limiting + cost tracking.
"""
from __future__ import annotations
import os, json, asyncio, time, hashlib, sqlite3
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Tenant Registry ───────────────────────────────────────────────────────────

ROLES = {
    "admin": {"analyze", "finetune", "export", "view_costs", "manage_users"},
    "compliance_officer": {"analyze", "export", "view_costs"},
    "analyst": {"analyze"},
    "viewer": {},
}

@dataclass
class Tenant:
    tenant_id: str
    name: str
    plan: str        # "starter" | "pro" | "enterprise"
    rate_limit_rpm: int    # requests per minute
    cost_budget_usd: float # monthly budget
    users: dict[str, str]  # user_id -> role

TENANTS: dict[str, Tenant] = {
    "tenant_acme": Tenant("tenant_acme", "Acme Corp", "enterprise", 120, 500.0,
                          {"alice": "admin", "bob": "compliance_officer"}),
    "tenant_beta": Tenant("tenant_beta", "Beta Inc", "pro", 60, 100.0,
                          {"carol": "analyst"}),
    "tenant_demo": Tenant("tenant_demo", "Demo LLC", "starter", 10, 10.0,
                          {"dave": "viewer"}),
}


# ── RBAC ─────────────────────────────────────────────────────────────────────

def check_permission(tenant_id: str, user_id: str, action: str) -> bool:
    tenant = TENANTS.get(tenant_id)
    if not tenant:
        return False
    role = tenant.users.get(user_id)
    if not role:
        return False
    allowed = ROLES.get(role, set())
    return action in allowed


# ── Rate Limiter (Redis-backed, with in-memory fallback) ─────────────────────

_memory_rate_store: dict[str, list[float]] = {}  # fallback when Redis unavailable

class RateLimiter:
    def __init__(self):
        self._redis = None
        try:
            import redis  # type: ignore
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_connect_timeout=1,
            )
            self._redis.ping()
            print("  RateLimiter: connected to Redis")
        except Exception:
            print("  RateLimiter: Redis unavailable, using in-memory fallback")
            self._redis = None

    def is_allowed(self, tenant_id: str, rate_limit_rpm: int) -> tuple[bool, int]:
        """Token bucket — returns (allowed, remaining_tokens)."""
        key = f"rate:{tenant_id}"
        now = time.time()
        window = 60.0  # 1-minute sliding window

        if self._redis:
            # Redis sliding window: ZREMRANGEBYSCORE + ZADD + ZCARD
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zadd(key, {str(uuid_ts := f"{now}:{id(now)}"): now})
            pipe.zcard(key)
            pipe.expire(key, 120)
            _, _, count, _ = pipe.execute()
            remaining = max(0, rate_limit_rpm - count)
            return count <= rate_limit_rpm, remaining
        else:
            # In-memory fallback
            timestamps = _memory_rate_store.get(key, [])
            timestamps = [t for t in timestamps if t > now - window]
            count = len(timestamps) + 1
            if count <= rate_limit_rpm:
                timestamps.append(now)
                _memory_rate_store[key] = timestamps
                return True, rate_limit_rpm - count
            return False, 0


# ── Cost Tracker (SQLite) ─────────────────────────────────────────────────────

class CostTracker:
    def __init__(self, db_path: str = "/tmp/tenant_costs.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                ts REAL DEFAULT (unixepoch())
            )
        """)
        self._conn.commit()

    def record(self, tenant_id: str, user_id: str, model: str,
               input_tokens: int, output_tokens: int, cost_usd: float):
        self._conn.execute(
            "INSERT INTO usage (tenant_id, user_id, model, input_tokens, output_tokens, cost_usd) "
            "VALUES (?,?,?,?,?,?)",
            (tenant_id, user_id, model, input_tokens, output_tokens, cost_usd),
        )
        self._conn.commit()

    def get_monthly_cost(self, tenant_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage "
            "WHERE tenant_id=? AND ts > (unixepoch() - 30*86400)",
            (tenant_id,),
        ).fetchone()
        return row[0] if row else 0.0

    def get_breakdown(self, tenant_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT user_id, model, SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) "
            "FROM usage WHERE tenant_id=? GROUP BY user_id, model",
            (tenant_id,),
        ).fetchall()
        return [{"user": r[0], "model": r[1], "in_tok": r[2], "out_tok": r[3], "cost": r[4]}
                for r in rows]


# ── Namespace Isolation ───────────────────────────────────────────────────────

class TenantNamespace:
    """
    Isolates conversation history, memory, and vector store per tenant.
    Thread IDs are prefixed with tenant_id so no cross-tenant leakage.
    """
    def __init__(self):
        self._store: dict[str, list[dict]] = {}  # "tenant:user" -> messages

    def get_history(self, tenant_id: str, user_id: str) -> list[dict]:
        key = f"{tenant_id}:{user_id}"
        return list(self._store.get(key, []))

    def append(self, tenant_id: str, user_id: str, role: str, content: str):
        key = f"{tenant_id}:{user_id}"
        if key not in self._store:
            self._store[key] = []
        self._store[key].append({"role": role, "content": content})

    def clear(self, tenant_id: str, user_id: str):
        key = f"{tenant_id}:{user_id}"
        self._store.pop(key, None)


# ── Multi-Tenant Agent ────────────────────────────────────────────────────────

rate_limiter = RateLimiter()
cost_tracker = CostTracker()
namespace = TenantNamespace()

COST_PER_1K = {"openai/gpt-4o-mini": (0.00015, 0.0006), "openai/gpt-4o": (0.005, 0.015)}

async def tenant_agent(
    tenant_id: str, user_id: str, message: str,
    model: str = "openai/gpt-4o-mini",
) -> dict:
    """Multi-tenant compliance agent with isolation, RBAC, rate limiting, and cost tracking."""
    # 1. Resolve tenant
    tenant = TENANTS.get(tenant_id)
    if not tenant:
        return {"error": f"Unknown tenant: {tenant_id}"}

    # 2. RBAC check
    if not check_permission(tenant_id, user_id, "analyze"):
        role = tenant.users.get(user_id, "(unknown)")
        return {"error": f"User '{user_id}' (role={role}) does not have 'analyze' permission"}

    # 3. Rate limit check
    allowed, remaining = rate_limiter.is_allowed(tenant_id, tenant.rate_limit_rpm)
    if not allowed:
        return {"error": f"Rate limit exceeded ({tenant.rate_limit_rpm} rpm). Try again in a minute.",
                "remaining": 0}

    # 4. Budget check
    monthly_cost = cost_tracker.get_monthly_cost(tenant_id)
    if monthly_cost >= tenant.cost_budget_usd:
        return {"error": f"Monthly cost budget exhausted (${monthly_cost:.2f} / ${tenant.cost_budget_usd:.2f})"}

    # 5. Namespaced conversation history
    history = namespace.get_history(tenant_id, user_id)
    system = (f"You are a compliance assistant for {tenant.name}. "
              "Provide precise, actionable compliance guidance. "
              "Do NOT reference data from other organisations.")

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    namespace.append(tenant_id, user_id, "user", message)

    # 6. LLM call
    resp = await litellm.acompletion(model=model, messages=messages, temperature=0.2)
    reply = resp.choices[0].message.content
    namespace.append(tenant_id, user_id, "assistant", reply)

    # 7. Record cost
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    in_rate, out_rate = COST_PER_1K.get(model, (0.00015, 0.0006))
    cost = (in_tok / 1000 * in_rate) + (out_tok / 1000 * out_rate)
    cost_tracker.record(tenant_id, user_id, model, in_tok, out_tok, cost)

    return {
        "tenant": tenant.name,
        "user": user_id,
        "reply": reply,
        "rate_remaining": remaining,
        "monthly_cost_usd": monthly_cost + cost,
        "budget_usd": tenant.cost_budget_usd,
    }


# ── FastAPI App ───────────────────────────────────────────────────────────────

def create_multitenant_app():
    from fastapi import FastAPI, HTTPException, Header  # type: ignore
    from pydantic import BaseModel as PM  # type: ignore

    app = FastAPI(title="Multi-Tenant Compliance Agent")

    class ChatRequest(PM):
        tenant_id: str
        user_id: str
        message: str
        model: str = "openai/gpt-4o-mini"

    @app.post("/chat")
    async def chat(req: ChatRequest):
        result = await tenant_agent(req.tenant_id, req.user_id, req.message, req.model)
        if "error" in result:
            raise HTTPException(403, result["error"])
        return result

    @app.get("/costs/{tenant_id}")
    async def tenant_costs(tenant_id: str, user_id: str):
        if not check_permission(tenant_id, user_id, "view_costs"):
            raise HTTPException(403, "Insufficient permissions")
        breakdown = cost_tracker.get_breakdown(tenant_id)
        monthly = cost_tracker.get_monthly_cost(tenant_id)
        budget = TENANTS[tenant_id].cost_budget_usd
        return {"monthly_cost_usd": monthly, "budget_usd": budget, "breakdown": breakdown}

    return app


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 33: Multi-Tenant Agent SOLUTION ===\n")

    print("1. Tenant Registry:")
    for tid, t in TENANTS.items():
        print(f"  {t.name} ({t.plan}): {t.rate_limit_rpm} rpm, ${t.cost_budget_usd}/mo budget")

    print("\n2. RBAC checks:")
    cases = [
        ("tenant_acme", "alice", "analyze"),
        ("tenant_acme", "alice", "finetune"),
        ("tenant_acme", "bob", "finetune"),
        ("tenant_beta", "carol", "export"),
        ("tenant_demo", "dave", "analyze"),
    ]
    for tid, uid, action in cases:
        ok = check_permission(tid, uid, action)
        role = TENANTS[tid].users.get(uid, "?")
        print(f"  {uid} ({role}) → {action}: {'✅' if ok else '❌'}")

    print("\n3. Running multi-tenant agent (isolation test):")
    # Alice from Acme asks about their contract
    r1 = await tenant_agent("tenant_acme", "alice",
                            "What are the key GDPR requirements for our data processing agreements?")
    if "error" not in r1:
        print(f"  Acme/alice: {r1['reply'][:150]}...")
        print(f"  Cost: ${r1['monthly_cost_usd']:.6f} / ${r1['budget_usd']:.2f}")
    else:
        print(f"  Acme/alice ERROR: {r1['error']}")

    # Carol from Beta asks — separate namespace, no Acme data
    r2 = await tenant_agent("tenant_beta", "carol",
                            "Summarize key SOC2 controls we need.")
    if "error" not in r2:
        print(f"  Beta/carol: {r2['reply'][:150]}...")
    else:
        print(f"  Beta/carol ERROR: {r2['error']}")

    # Dave (viewer) tries to call agent — should be rejected
    r3 = await tenant_agent("tenant_demo", "dave", "Give me compliance advice.")
    print(f"  Demo/dave (viewer): {'❌ Blocked as expected' if 'error' in r3 else '✅ ' + r3['reply'][:80]}")

    print("\n4. Rate limiting — burst test on tenant_demo (10 rpm limit):")
    for i in range(3):
        allowed, rem = rate_limiter.is_allowed("tenant_demo", TENANTS["tenant_demo"].rate_limit_rpm)
        print(f"  Request {i+1}: allowed={allowed}, remaining={rem}")

    print("\n5. Cost breakdown — Acme:")
    for row in cost_tracker.get_breakdown("tenant_acme"):
        print(f"  {row['user']} | {row['model']} | ${row['cost']:.6f}")

    print("\n6. To run as FastAPI server:")
    print("   app = create_multitenant_app()")
    print("   uvicorn.run(app, host='0.0.0.0', port=8000)")
    print("   POST /chat           — send message as tenant user")
    print("   GET  /costs/{tid}    — view tenant cost breakdown")

if __name__ == "__main__":
    asyncio.run(main())
