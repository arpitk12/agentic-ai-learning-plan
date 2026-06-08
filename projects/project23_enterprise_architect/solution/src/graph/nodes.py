"""LangGraph node functions for the compliance review workflow."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from langgraph.types import interrupt

from src.agents.policy_agent import check_document_policies
from src.agents.risk_agent import assess_document_risk
from src.config import cfg
from src.graph.state import ComplianceState


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Node 1: intake ────────────────────────────────────────────────────────

async def intake_node(state: ComplianceState) -> dict:
    """Fetch document from MCP server and validate basic structure."""
    start = time.time()
    # In production: use MCP client to fetch from document server
    # For standalone: content is passed directly in state
    content = state.get("document_content") or f"[Document {state['document_id']} fetched via MCP]"

    if not content or len(content.strip()) < 50:
        return {
            "status": "intake_failed",
            "audit_entries": [{"step": "intake", "ts": _ts(), "error": "Document too short or empty"}],
        }

    return {
        "document_content": content,
        "status": "intake_complete",
        "processing_time_seconds": time.time() - start,
        "llm_cost_usd": 0.0,
        "audit_entries": [{"step": "intake", "ts": _ts(), "doc_id": state["document_id"], "actor": "system"}],
        "messages": [{"role": "system", "content": f"Intake complete: {state['document_id']}"}],
    }


# ── Node 2: risk_analysis ─────────────────────────────────────────────────

async def risk_analysis_node(state: ComplianceState) -> dict:
    """PydanticAI risk agent assesses regulatory and legal risk."""
    assessment, cost = await assess_document_risk(
        state["document_content"],
        state.get("document_type", "unknown"),
    )
    return {
        "risk_assessment": assessment.model_dump(),
        "llm_cost_usd": state.get("llm_cost_usd", 0.0) + cost,
        "status": "risk_assessed",
        "audit_entries": [{
            "step": "risk_analysis",
            "ts": _ts(),
            "risk_level": assessment.risk_level,
            "confidence": assessment.confidence,
            "actor": "risk-agent",
        }],
        "messages": [{"role": "assistant", "content": f"Risk: {assessment.risk_level} ({assessment.confidence:.0%} confidence)"}],
    }


# ── Node 3: policy_check ──────────────────────────────────────────────────

async def policy_check_node(state: ComplianceState) -> dict:
    """PydanticAI policy agent checks document against company policies."""
    # Load policies — in production, fetched from MCP policy server
    policies = cfg.default_policies

    policy_result, cost = await check_document_policies(
        document_content=state["document_content"],
        policies=policies,
        document_type=state.get("document_type", "unknown"),
    )
    return {
        "policy_check": policy_result.model_dump(),
        "compliance_score": policy_result.compliance_score,
        "llm_cost_usd": state.get("llm_cost_usd", 0.0) + cost,
        "status": "policy_checked",
        "audit_entries": [{
            "step": "policy_check",
            "ts": _ts(),
            "is_compliant": policy_result.is_compliant,
            "score": policy_result.compliance_score,
            "violations": len(policy_result.violations),
            "actor": "policy-agent",
        }],
    }


# ── Node 4: human_review (HITL) ───────────────────────────────────────────

async def human_review_node(state: ComplianceState) -> dict:
    """Pause for human review. Resumes via Command(resume=<feedback>)."""
    risk = state.get("risk_assessment", {})
    violations = state.get("policy_check", {}).get("violations", [])

    feedback = interrupt({
        "document_id": state["document_id"],
        "risk_level": risk.get("risk_level"),
        "risk_factors": risk.get("risk_factors", []),
        "policy_violations": len(violations),
        "compliance_score": state.get("compliance_score", 0),
        "prompt": "Respond: 'approve: <notes>' | 'reject: <notes>' | 'escalate: <notes>'",
    })

    text = str(feedback).strip()
    action, _, notes = text.partition(":")
    action = action.strip().lower()

    status_map = {"approve": "approved", "reject": "rejected", "escalate": "escalated"}
    return {
        "human_feedback": notes.strip(),
        "human_reviewer": "human",
        "escalated_to_legal": action == "escalate",
        "status": status_map.get(action, "pending_review"),
        "audit_entries": [{
            "step": "human_review",
            "ts": _ts(),
            "action": action,
            "notes": notes.strip(),
            "actor": "human-reviewer",
        }],
    }


# ── Node 5: auto_approve ──────────────────────────────────────────────────

async def auto_approve_node(state: ComplianceState) -> dict:
    """Auto-approve low/medium risk compliant documents."""
    return {
        "status": "approved",
        "audit_entries": [{"step": "auto_approve", "ts": _ts(), "actor": "system"}],
    }


# ── Node 6: audit_log ────────────────────────────────────────────────────

async def audit_log_node(state: ComplianceState) -> dict:
    """Write final audit trail to MCP audit server."""
    # In production: call MCP audit server write_audit_entry tool
    # For standalone: entries already accumulated in state["audit_entries"]
    return {
        "audit_entries": [{"step": "audit_finalized", "ts": _ts(), "actor": "system",
                           "total_entries": len(state.get("audit_entries", []))}],
        "messages": [{"role": "system", "content": "Audit trail finalized."}],
    }


# ── Node 7: report ────────────────────────────────────────────────────────

async def report_node(state: ComplianceState) -> dict:
    """Generate final compliance report."""
    manual_cost = 5.0 * 85.0  # 5 analyst hours × $85/hr
    return {
        "status": state.get("status", "completed"),
        "audit_entries": [{
            "step": "report_generated",
            "ts": _ts(),
            "actor": "system",
            "llm_cost_usd": state.get("llm_cost_usd", 0),
            "manual_cost_equivalent_usd": manual_cost,
            "savings_usd": manual_cost - state.get("llm_cost_usd", 0),
        }],
    }


# ── Conditional edge function ────────────────────────────────────────────

def route_by_risk(state: ComplianceState) -> str:
    """Route to human review for high/critical risk, auto-approve otherwise."""
    risk = state.get("risk_assessment", {})
    policy = state.get("policy_check", {})
    level = risk.get("risk_level", "low")
    is_compliant = policy.get("is_compliant", True)

    if level in ("high", "critical") or not is_compliant:
        return "human_review"
    return "auto_approve"
