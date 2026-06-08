"""
Project 32 SOLUTION — Agent-to-Agent (A2A) Protocol
Google A2A protocol: agent cards + task delegation + JWT auth + 3-agent chain.
"""
from __future__ import annotations
import os, json, asyncio, uuid, time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()


# ── Agent Cards ───────────────────────────────────────────────────────────────

def make_agent_card(
    name: str, description: str, url: str,
    capabilities: list[str], input_schema: dict, output_schema: dict,
) -> dict:
    return {
        "name": name,
        "description": description,
        "url": url,
        "version": "1.0",
        "capabilities": capabilities,
        "authentication": {"schemes": ["bearer_jwt"]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [{
            "id": name.lower().replace(" ", "_"),
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "outputSchema": output_schema,
        }],
    }

AGENT_CARDS = {
    "extractor": make_agent_card(
        "Document Extractor",
        "Extracts entities, key clauses, and metadata from business documents",
        "http://localhost:8001",
        ["entity_extraction", "clause_identification"],
        {"type": "object", "properties": {"document": {"type": "string"}, "doc_type": {"type": "string"}}},
        {"type": "object", "properties": {"entities": {"type": "array"}, "clauses": {"type": "array"}}},
    ),
    "analyzer": make_agent_card(
        "Compliance Analyzer",
        "Analyzes extracted data and classifies compliance risk with reasoning",
        "http://localhost:8002",
        ["risk_classification", "regulatory_mapping"],
        {"type": "object", "properties": {"entities": {"type": "array"}, "clauses": {"type": "array"}}},
        {"type": "object", "properties": {"risk_level": {"type": "string"}, "findings": {"type": "array"}}},
    ),
    "reporter": make_agent_card(
        "Report Generator",
        "Generates human-readable compliance reports from analysis results",
        "http://localhost:8003",
        ["report_generation", "recommendation"],
        {"type": "object", "properties": {"risk_level": {"type": "string"}, "findings": {"type": "array"}}},
        {"type": "object", "properties": {"report": {"type": "string"}, "recommendations": {"type": "array"}}},
    ),
}


# ── JWT Auth ──────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("A2A_JWT_SECRET", "dev-secret-change-in-production")

def create_token(agent_id: str, target_agent: str) -> str:
    try:
        import jwt  # type: ignore
        payload = {
            "sub": agent_id,
            "aud": target_agent,
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,   # 5-minute expiry
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    except ImportError:
        # Fallback: simple base64 "token" (not secure — install pyjwt for production)
        import base64
        payload = json.dumps({"sub": agent_id, "aud": target_agent})
        return base64.b64encode(payload.encode()).decode()

def verify_token(token: str, expected_audience: str) -> dict | None:
    try:
        import jwt  # type: ignore
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=expected_audience)
    except ImportError:
        import base64
        try:
            payload = json.loads(base64.b64decode(token).decode())
            return payload if payload.get("aud") == expected_audience else None
        except Exception:
            return None
    except Exception:
        return None


# ── Task Protocol ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "submitted"   # submitted | working | completed | failed
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

_task_store: dict[str, Task] = {}


# ── Agent Implementations ─────────────────────────────────────────────────────

async def extractor_agent(document: str, doc_type: str, auth_token: str) -> dict:
    """Agent 1: Extracts entities and key clauses from a document."""
    claims = verify_token(auth_token, "extractor_agent")
    if not claims:
        raise PermissionError("Invalid or expired token")

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Extract from this {doc_type} document:
1. Organizations mentioned (with their roles)
2. Key compliance clauses present or missing
3. Dollar amounts and dates

Document: {document}

Return JSON: {{"entities": [{{"name": str, "role": str}}], "clauses": [{{"clause": str, "present": bool}}], "amounts": [str]}}""",
        }],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)

async def analyzer_agent(entities: list, clauses: list, auth_token: str) -> dict:
    """Agent 2: Classifies risk from extracted data."""
    claims = verify_token(auth_token, "analyzer_agent")
    if not claims:
        raise PermissionError("Invalid or expired token")

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Analyze compliance risk from:
Entities: {json.dumps(entities)}
Clauses: {json.dumps(clauses)}

Return JSON: {{"risk_level": "low|medium|high|critical", "findings": [{{"issue": str, "regulation": str, "severity": str}}]}}""",
        }],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)

async def reporter_agent(risk_level: str, findings: list, auth_token: str) -> dict:
    """Agent 3: Generates final compliance report."""
    claims = verify_token(auth_token, "reporter_agent")
    if not claims:
        raise PermissionError("Invalid or expired token")

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate a compliance report:
Risk Level: {risk_level}
Findings: {json.dumps(findings)}

Return JSON: {{"report": "executive summary (2-3 paragraphs)", "recommendations": [{{"action": str, "priority": str, "deadline": str}}]}}""",
        }],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


# ── Orchestrator: 3-Agent Chain ───────────────────────────────────────────────

async def orchestrate_compliance_review(document: str, doc_type: str) -> dict:
    """Orchestrate a 3-agent pipeline: Extract → Analyze → Report."""
    orchestrator_id = "orchestrator"

    # Step 1: Call extractor agent
    ext_token = create_token(orchestrator_id, "extractor_agent")
    print("  [A2A] → Calling Extractor Agent...")
    extracted = await extractor_agent(document, doc_type, ext_token)
    print(f"  [A2A] ✓ Extractor: {len(extracted.get('entities', []))} entities, "
          f"{len(extracted.get('clauses', []))} clauses")

    # Step 2: Call analyzer agent with extractor's output
    ana_token = create_token(orchestrator_id, "analyzer_agent")
    print("  [A2A] → Calling Analyzer Agent...")
    analysis = await analyzer_agent(
        extracted.get("entities", []),
        extracted.get("clauses", []),
        ana_token,
    )
    print(f"  [A2A] ✓ Analyzer: risk={analysis.get('risk_level')} "
          f"findings={len(analysis.get('findings', []))}")

    # Step 3: Call reporter agent with analyzer's output
    rep_token = create_token(orchestrator_id, "reporter_agent")
    print("  [A2A] → Calling Reporter Agent...")
    report = await reporter_agent(
        analysis.get("risk_level", "unknown"),
        analysis.get("findings", []),
        rep_token,
    )
    print(f"  [A2A] ✓ Reporter: {len(report.get('recommendations', []))} recommendations")

    return {
        "pipeline": "extractor → analyzer → reporter",
        "extracted": extracted,
        "analysis": analysis,
        "report": report,
    }


# ── FastAPI A2A Server ────────────────────────────────────────────────────────

def create_a2a_app():
    from fastapi import FastAPI, HTTPException, Header  # type: ignore
    from pydantic import BaseModel as PM

    app = FastAPI(title="A2A Compliance Orchestrator")

    class ReviewRequest(PM):
        document: str
        doc_type: str = "contract"

    @app.get("/.well-known/agent.json")
    async def agent_card():
        return AGENT_CARDS["extractor"]

    @app.post("/tasks")
    async def create_task(req: ReviewRequest, authorization: str = Header(None)):
        task = Task()
        _task_store[task.task_id] = task
        asyncio.create_task(_run_review(task.task_id, req.document, req.doc_type))
        return {"task_id": task.task_id, "status": task.status}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        if task_id not in _task_store:
            raise HTTPException(404, "Task not found")
        t = _task_store[task_id]
        return {"task_id": task_id, "status": t.status, "result": t.result, "error": t.error}

    async def _run_review(task_id: str, document: str, doc_type: str):
        task = _task_store[task_id]
        task.status = "working"
        try:
            result = await orchestrate_compliance_review(document, doc_type)
            task.result = result
            task.status = "completed"
        except Exception as e:
            task.error = str(e)
            task.status = "failed"

    return app


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 32: A2A Protocol SOLUTION ===\n")

    print("1. Agent Cards (Discovery):")
    for agent_name, card in AGENT_CARDS.items():
        print(f"  {card['name']}: {card['url']} | capabilities={card['capabilities']}")

    print("\n2. JWT Authentication test:")
    token = create_token("orchestrator", "extractor_agent")
    claims = verify_token(token, "extractor_agent")
    print(f"  Token created: {token[:30]}...")
    print(f"  Token verified: sub={claims.get('sub')} aud={claims.get('aud')}")

    print("\n3. Running 3-agent pipeline:")
    result = await orchestrate_compliance_review(
        document="Cloud services agreement between Acme Corp and AWS for EU data processing. "
                 "No Data Processing Agreement (DPA) is attached. Annual contract value: $1.2M. "
                 "Governed by GDPR Article 28. Signed March 2026.",
        doc_type="contract",
    )

    print(f"\n4. Final Report:")
    report = result["report"]
    print(f"  Risk Level: {result['analysis']['risk_level']}")
    print(f"  Report: {report.get('report', '')[:250]}...")
    print(f"  Recommendations ({len(report.get('recommendations', []))}):")
    for rec in report.get("recommendations", [])[:3]:
        print(f"    [{rec.get('priority', '?').upper()}] {rec.get('action', '')}")

    print("\n5. To run as FastAPI server:")
    print("   app = create_a2a_app()")
    print("   uvicorn.run(app, host='0.0.0.0', port=8000)")
    print("   POST /tasks — submit document for review")
    print("   GET  /tasks/{id} — poll for result")
    print("   GET  /.well-known/agent.json — A2A agent card discovery")

if __name__ == "__main__":
    asyncio.run(main())
