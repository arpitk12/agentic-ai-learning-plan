# Project 23 Guide — Enterprise Compliance Review Agent

> Stack: **LangGraph + LangChain · PydanticAI · MCP · Langfuse/LangSmith · AWS AgentCore / Vertex AI**

---

## Phase 1 — PydanticAI Contracts (Typed Agent Interfaces)

PydanticAI enforces type safety at the agent boundary — you get compile-time errors for malformed outputs instead of runtime JSON parsing surprises.

### 1.1 Core Concept

```python
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Literal

# The contract: what the agent MUST return
class RiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high", "critical"]
    risk_factors: list[str] = Field(min_length=1)
    regulatory_concerns: list[str]
    requires_human_review: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

# The typed agent — result_type enforces the contract
risk_agent = Agent(
    model="openai:gpt-4o-mini",
    result_type=RiskAssessment,
    system_prompt="""You are a compliance risk analyst. Analyze documents for:
    - GDPR / data protection risks
    - Financial regulation compliance (SOX, PCI-DSS)
    - Contract liability exposure
    Always return structured, justified risk assessments.""",
)

# Run it — result.data is a validated RiskAssessment, not raw JSON
async def assess_risk(document: str) -> RiskAssessment:
    result = await risk_agent.run(document)
    return result.data   # type: RiskAssessment — guaranteed by PydanticAI
```

### 1.2 Dependency Injection with RunContext

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class ComplianceDeps:
    company_policies: dict[str, str]
    regulatory_db: dict         # loaded from MCP at startup
    reviewer_name: str

policy_agent = Agent(
    model="openai:gpt-4o-mini",
    result_type=PolicyCheckResult,
    deps_type=ComplianceDeps,
)

@policy_agent.system_prompt
async def build_system_prompt(ctx: RunContext[ComplianceDeps]) -> str:
    policy_count = len(ctx.deps.company_policies)
    return f"You enforce {policy_count} company policies. Reviewer: {ctx.deps.reviewer_name}."

@policy_agent.tool
async def lookup_policy(ctx: RunContext[ComplianceDeps], policy_id: str) -> str:
    """Look up a specific company policy by ID."""
    return ctx.deps.company_policies.get(policy_id, "Policy not found")

# Usage
deps = ComplianceDeps(
    company_policies={"POL-001": "No PII in logs", "POL-002": "Encrypt at rest"},
    regulatory_db={},
    reviewer_name="Alice Chen",
)
result = await policy_agent.run(document_text, deps=deps)
```

### 1.3 ModelRetry — Self-Healing Agents

```python
from pydantic_ai import ModelRetry

@risk_agent.result_validator
async def validate_risk_result(ctx: RunContext, result: RiskAssessment) -> RiskAssessment:
    """Force the agent to retry if confidence is too low."""
    if result.confidence < 0.7 and result.risk_level == "critical":
        raise ModelRetry(
            "Confidence too low for a CRITICAL assessment. "
            "Provide more specific regulatory citations and retry."
        )
    return result
```

---

## Phase 2 — LangGraph Multi-Agent Workflow

### 2.1 Compliance State

```python
# src/graph/state.py
import operator
from typing import Annotated
from typing_extensions import TypedDict

class ComplianceState(TypedDict):
    # Document
    document_id: str
    document_content: str
    document_type: str           # "contract" | "policy" | "data-processing" | "vendor"

    # Analysis results
    risk_assessment: dict        # RiskAssessment serialized
    policy_violations: list[str]
    compliance_score: float      # 0-100

    # Human-in-the-loop
    human_feedback: str | None
    escalated_to_legal: bool

    # Audit trail (appended, never overwritten — reducer pattern)
    audit_entries: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]

    # Cost tracking
    llm_cost_usd: float
    processing_time_seconds: float

    # Status
    status: str    # intake|risk_assessed|policy_checked|approved|rejected|escalated
```

### 2.2 Node Implementation Pattern

```python
# src/graph/nodes.py
import time
from src.agents.risk_agent import risk_agent
from src.graph.state import ComplianceState
from langgraph.types import interrupt

async def intake_node(state: ComplianceState) -> dict:
    """Validate document exists and is parseable."""
    start = time.time()
    # MCP call to fetch document
    content = await fetch_document_via_mcp(state["document_id"])
    return {
        "document_content": content,
        "status": "intake_complete",
        "processing_time_seconds": time.time() - start,
        "audit_entries": [{"step": "intake", "ts": time.time(), "doc_id": state["document_id"]}],
    }

async def risk_analysis_node(state: ComplianceState) -> dict:
    """PydanticAI risk agent assesses the document."""
    result = await risk_agent.run(state["document_content"])
    assessment = result.data

    # Track LLM cost
    cost = result.usage().total_tokens * 0.00000015  # gpt-4o-mini rate

    return {
        "risk_assessment": assessment.model_dump(),
        "llm_cost_usd": state.get("llm_cost_usd", 0) + cost,
        "status": "risk_assessed",
        "audit_entries": [{"step": "risk_analysis", "result": assessment.risk_level}],
    }

async def human_review_node(state: ComplianceState) -> dict:
    """Pause for human review on HIGH/CRITICAL risk documents."""
    assessment = state["risk_assessment"]
    feedback = interrupt({
        "document_id": state["document_id"],
        "risk_level": assessment["risk_level"],
        "risk_factors": assessment["risk_factors"],
        "prompt": "Approve, reject, or escalate with notes. Format: 'approve|reject|escalate: <notes>'",
    })
    action, _, notes = str(feedback).partition(":")
    action = action.strip().lower()

    status_map = {"approve": "approved", "reject": "rejected", "escalate": "escalated"}
    return {
        "human_feedback": notes.strip(),
        "escalated_to_legal": action == "escalate",
        "status": status_map.get(action, "pending_review"),
        "audit_entries": [{"step": "human_review", "action": action, "reviewer_notes": notes}],
    }

def route_by_risk(state: ComplianceState) -> str:
    """Conditional edge — route based on risk level."""
    level = state.get("risk_assessment", {}).get("risk_level", "low")
    if level in ("high", "critical"):
        return "human_review"
    return "audit_log"
```

### 2.3 Graph Assembly

```python
# src/graph/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph.state import ComplianceState
from src.graph.nodes import (
    intake_node, risk_analysis_node, policy_check_node,
    human_review_node, audit_log_node, report_node, route_by_risk,
)

def build_compliance_graph(checkpointer=None):
    builder = StateGraph(ComplianceState)

    builder.add_node("intake",        intake_node)
    builder.add_node("risk_analysis", risk_analysis_node)
    builder.add_node("policy_check",  policy_check_node)
    builder.add_node("human_review",  human_review_node)
    builder.add_node("audit_log",     audit_log_node)
    builder.add_node("report",        report_node)

    builder.add_edge(START,           "intake")
    builder.add_edge("intake",        "risk_analysis")
    builder.add_edge("risk_analysis", "policy_check")

    # Route: high/critical → HITL, else → auto-approve path
    builder.add_conditional_edges(
        "policy_check",
        route_by_risk,
        {"human_review": "human_review", "audit_log": "audit_log"},
    )

    builder.add_edge("human_review", "audit_log")
    builder.add_edge("audit_log",    "report")
    builder.add_edge("report",        END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
```

---

## Phase 3 — MCP Enterprise Servers

MCP (Model Context Protocol) decouples your agents from data sources. Each server is independently deployable, versionable, and auditable.

### 3.1 Document Store MCP Server

```python
# src/mcp/document_server.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("enterprise-document-store")

@mcp.tool()
async def get_document(doc_id: str, version: str = "latest") -> dict:
    """Retrieve document from enterprise document store (SharePoint/S3)."""
    # In production: connect to SharePoint, S3, or DMS
    return {
        "doc_id": doc_id,
        "content": f"[Document content for {doc_id}]",
        "version": version,
        "retrieved_at": datetime.utcnow().isoformat(),
    }

@mcp.tool()
async def list_pending_documents(doc_type: str = "all") -> list[dict]:
    """List documents awaiting compliance review."""
    return []  # Connect to your queue/DMS

@mcp.resource("documents://schema")
async def document_schema() -> str:
    return "Supported types: contract, policy, data-processing, vendor-agreement"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3.2 Audit Log MCP Server

```python
# src/mcp/audit_server.py
from mcp.server.fastmcp import FastMCP
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path

mcp = FastMCP("compliance-audit-log")
AUDIT_FILE = Path("data/audit_trail.jsonl")

@mcp.tool()
async def write_audit_entry(
    document_id: str,
    step: str,
    action: str,
    actor: str,
    details: dict,
) -> dict:
    """Append an immutable audit entry. Hash-chained for tamper evidence."""
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Build entry with hash chain
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "doc_id": document_id,
        "step": step,
        "action": action,
        "actor": actor,
        "details": details,
    }

    # Tamper-evident: hash of previous entry + current content
    prev_hash = _get_last_hash()
    entry["prev_hash"] = prev_hash
    entry["hash"] = hashlib.sha256(
        (prev_hash + json.dumps(entry, sort_keys=True)).encode()
    ).hexdigest()

    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "written", "hash": entry["hash"]}

@mcp.tool()
async def get_audit_trail(document_id: str) -> list[dict]:
    """Retrieve full audit trail for a document."""
    if not AUDIT_FILE.exists():
        return []
    entries = [json.loads(line) for line in AUDIT_FILE.read_text().splitlines()]
    return [e for e in entries if e.get("doc_id") == document_id]

def _get_last_hash() -> str:
    if not AUDIT_FILE.exists():
        return "GENESIS"
    lines = AUDIT_FILE.read_text().splitlines()
    return json.loads(lines[-1])["hash"] if lines else "GENESIS"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## Phase 4 — Observability: Langfuse + LangSmith

### 4.1 Langfuse — Cost Tracking and Traces

```python
# src/observability/langfuse_tracer.py
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context, observe

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

def get_callback(session_id: str, user_id: str, doc_id: str) -> CallbackHandler:
    """LangChain-compatible callback that sends traces to Langfuse."""
    return CallbackHandler(
        session_id=session_id,
        user_id=user_id,
        metadata={"document_id": doc_id, "project": "compliance-review"},
        tags=["compliance", "automated-review"],
    )

@observe(name="compliance-review-run")
async def traced_compliance_run(graph, state: dict, config: dict) -> dict:
    """Run compliance graph with full Langfuse tracing."""
    langfuse_context.update_current_observation(
        input={"document_id": state["document_id"]},
        metadata={"document_type": state.get("document_type")},
    )
    result = await graph.ainvoke(state, config=config)
    langfuse_context.update_current_observation(
        output={"status": result.get("status"), "cost_usd": result.get("llm_cost_usd")},
    )
    return result
```

### 4.2 Cost Dashboard Query

```python
# Get cost per document type from Langfuse SDK
def get_cost_report(days: int = 30) -> dict:
    traces = langfuse.get_traces(tags=["compliance"], limit=500)
    by_type = {}
    for trace in traces.data:
        doc_type = (trace.metadata or {}).get("document_type", "unknown")
        cost = sum(g.calculated_total_cost or 0 for g in trace.observations or [])
        by_type.setdefault(doc_type, []).append(cost)
    return {t: {"count": len(v), "avg_cost": sum(v)/len(v), "total": sum(v)}
            for t, v in by_type.items()}
```

### 4.3 LangSmith — Evaluation Dataset

```python
# src/observability/langsmith_tracer.py
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

client = Client()

@traceable(name="risk-analysis", project_name="compliance-review")
async def traced_risk_analysis(document: str) -> dict:
    """Automatically traced by LangSmith."""
    result = await risk_agent.run(document)
    return result.data.model_dump()

def create_eval_dataset(examples: list[dict]) -> None:
    """Create a golden dataset for regression testing."""
    client.create_dataset(
        dataset_name="compliance-review-golden",
        description="Golden examples for compliance review evaluation",
    )
    client.create_examples(
        inputs=[{"document": e["doc"]} for e in examples],
        outputs=[{"risk_level": e["expected_risk"]} for e in examples],
        dataset_name="compliance-review-golden",
    )

def run_evaluation() -> None:
    """Run automated evaluation against golden dataset."""
    def risk_accuracy(run, example) -> dict:
        predicted = run.outputs.get("risk_level")
        expected = example.outputs.get("risk_level")
        return {"score": 1 if predicted == expected else 0}

    evaluate(
        traced_risk_analysis,
        data="compliance-review-golden",
        evaluators=[risk_accuracy],
        experiment_prefix="compliance-risk-",
    )
```

---

## Phase 5 — Cloud Runtime

### 5.1 AWS AgentCore

AWS AgentCore (Amazon Bedrock AgentCore) provides a managed, compliant runtime for agents with built-in IAM, VPC isolation, and CloudTrail logging.

```python
# src/runtime/aws_agentcore.py
import boto3, json, os

class AgentCoreRuntime:
    """Deploy and invoke the compliance agent via AWS AgentCore."""

    def __init__(self):
        self.client = boto3.client(
            "bedrock-agentcore",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.agent_id = os.environ.get("AGENTCORE_AGENT_ID")

    def invoke(self, document_id: str, document_type: str,
               session_id: str | None = None) -> dict:
        """Invoke the compliance agent via managed AgentCore runtime."""
        response = self.client.invoke_agent(
            agentId=self.agent_id,
            sessionId=session_id or document_id,
            inputText=json.dumps({"document_id": document_id, "document_type": document_type}),
        )
        # AgentCore handles: IAM auth, VPC isolation, CloudTrail logging,
        # memory management, and response streaming
        output = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                output += event["chunk"].get("bytes", b"").decode()
        return json.loads(output)

    def get_compliance_logs(self, document_id: str) -> list[dict]:
        """Retrieve CloudTrail audit logs for a document review."""
        cloudtrail = boto3.client("cloudtrail", region_name=os.environ.get("AWS_REGION"))
        events = cloudtrail.lookup_events(
            LookupAttributes=[{"AttributeKey": "ResourceName", "AttributeValue": document_id}]
        )
        return events.get("Events", [])
```

### 5.2 Google Vertex AI Agent Engine

```python
# src/runtime/vertex_ai.py
import os
from google.cloud import aiplatform
from vertexai.preview.reasoning_engines import ReasoningEngine

class VertexAIRuntime:
    """Deploy compliance agent to Vertex AI Agent Engine."""

    def __init__(self):
        aiplatform.init(
            project=os.environ["GCP_PROJECT_ID"],
            location=os.environ.get("GCP_REGION", "us-central1"),
        )

    def deploy_agent(self, agent_app) -> ReasoningEngine:
        """Package and deploy the LangGraph agent to Vertex AI."""
        return ReasoningEngine.create(
            agent_app,
            requirements=[
                "langchain-google-vertexai>=2.0",
                "langgraph>=0.2",
                "pydantic-ai>=0.1",
                "langfuse>=3.0",
            ],
            display_name="compliance-review-agent",
            description="Enterprise compliance document review — Q1 automation",
        )

    def invoke_deployed(self, engine: ReasoningEngine,
                        document_id: str, document_type: str) -> dict:
        """Invoke a deployed agent on Vertex AI."""
        return engine.query(
            input={"document_id": document_id, "document_type": document_type}
        )
```

---

## Phase 6 — Cost Reduction Measurement (Q1 Goal)

```python
# src/reporting/cost_report.py
from dataclasses import dataclass

@dataclass
class ProcessCost:
    """Cost model for before/after comparison."""
    manual_analyst_hours: float = 5.0      # hours per document
    analyst_hourly_rate: float = 85.0      # USD/hour (fully loaded)
    documents_per_month: int = 400
    llm_cost_per_doc: float = 3.50         # avg LLM + compute
    review_rate: float = 0.15              # 15% still need human review

    @property
    def manual_monthly_cost(self) -> float:
        return self.documents_per_month * self.manual_analyst_hours * self.analyst_hourly_rate

    @property
    def automated_monthly_cost(self) -> float:
        llm_cost = self.documents_per_month * self.llm_cost_per_doc
        human_review_cost = (
            self.documents_per_month * self.review_rate
            * (self.manual_analyst_hours * 0.3)   # review is 30% of full analysis time
            * self.analyst_hourly_rate
        )
        return llm_cost + human_review_cost

    @property
    def monthly_savings(self) -> float:
        return self.manual_monthly_cost - self.automated_monthly_cost

    @property
    def savings_percent(self) -> float:
        return (self.monthly_savings / self.manual_monthly_cost) * 100

    def print_report(self) -> None:
        print(f"""
╔══════════════════════════════════════════════════════╗
║           Q1 COST REDUCTION REPORT                   ║
╠══════════════════════════════════════════════════════╣
║ Documents/month:        {self.documents_per_month:>6}                  ║
║ Manual cost/month:      ${self.manual_monthly_cost:>10,.0f}            ║
║ Automated cost/month:   ${self.automated_monthly_cost:>10,.0f}         ║
║ Monthly savings:        ${self.monthly_savings:>10,.0f}                ║
║ Savings %:              {self.savings_percent:>9.1f}%                  ║
║ Annual savings:         ${self.monthly_savings * 12:>10,.0f}           ║
╚══════════════════════════════════════════════════════╝
        """)

# With defaults: ~$170k/month manual → ~$10k/month automated = 94% savings
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Typed contracts | PydanticAI | Catch malformed outputs at dev time, not runtime |
| Orchestration | LangGraph | Native interrupt/resume for HITL, SqliteSaver for audit |
| Data connectivity | MCP | Decoupled, independently deployable, versionable |
| Observability | Langfuse | Open-source, self-hostable, cost-per-run tracking |
| Compliance runtime | AWS AgentCore | CloudTrail + IAM + VPC — required for SOC2/HIPAA workloads |
| Analytics runtime | Vertex AI | BigQuery integration for document analytics at scale |
| Audit trail | Hash-chained JSONL | Tamper-evident, simple, no DB dependency |
