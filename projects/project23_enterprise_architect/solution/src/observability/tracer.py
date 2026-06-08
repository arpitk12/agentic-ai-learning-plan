"""Langfuse and LangSmith observability integrations."""
from __future__ import annotations

import os
from typing import Any

from src.config import cfg


# ── Langfuse ──────────────────────────────────────────────────────────────

def get_langfuse_handler(session_id: str, doc_id: str):
    """Return a LangChain-compatible Langfuse callback handler."""
    if not cfg.langfuse_public_key:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            host=cfg.langfuse_host,
            session_id=session_id,
            metadata={"document_id": doc_id, "project": "compliance-review"},
            tags=["compliance", "automated-review"],
        )
    except ImportError:
        return None


def get_langfuse_client():
    """Return a Langfuse client for manual tracing and cost queries."""
    if not cfg.langfuse_public_key:
        return None
    from langfuse import Langfuse
    return Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
        host=cfg.langfuse_host,
    )


def get_cost_report(days: int = 30) -> dict[str, Any]:
    """Query Langfuse for cost breakdown by document type."""
    client = get_langfuse_client()
    if not client:
        return {"error": "Langfuse not configured"}

    traces = client.get_traces(tags=["compliance"], limit=1000)
    by_type: dict[str, list[float]] = {}
    for trace in traces.data:
        doc_type = (trace.metadata or {}).get("document_type", "unknown")
        cost = sum(
            getattr(obs, "calculated_total_cost", None) or 0
            for obs in (trace.observations or [])
        )
        by_type.setdefault(doc_type, []).append(cost)

    return {
        doc_type: {
            "count": len(costs),
            "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0,
            "total_cost_usd": round(sum(costs), 4),
        }
        for doc_type, costs in by_type.items()
    }


# ── LangSmith ─────────────────────────────────────────────────────────────

def configure_langsmith() -> bool:
    """Enable LangSmith tracing via environment variables."""
    if not cfg.langsmith_api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
    return True


def create_golden_dataset(examples: list[dict]) -> None:
    """Upload a golden evaluation dataset to LangSmith."""
    if not cfg.langsmith_api_key:
        raise RuntimeError("LANGSMITH_API_KEY not set")
    from langsmith import Client
    client = Client()
    client.create_dataset(
        dataset_name="compliance-review-golden",
        description="Golden examples for compliance review regression testing",
    )
    client.create_examples(
        inputs=[{"document": e["document"], "doc_type": e["doc_type"]} for e in examples],
        outputs=[{"expected_risk_level": e["expected_risk_level"],
                  "expected_compliant": e["expected_compliant"]} for e in examples],
        dataset_name="compliance-review-golden",
    )
    print(f"Created dataset with {len(examples)} examples")
