# Project 23 — Enterprise Compliance Review Agent

A **production-grade, enterprise agentic system** that replaces a manual compliance review process with a multi-agent workflow. Built with the full enterprise architect stack: **LangGraph** orchestration, **LangChain** chains, **PydanticAI** contracts, **MCP** enterprise servers, **Langfuse/LangSmith** observability, and **AWS AgentCore / Google Vertex AI** as managed runtime.

**Q1 Goal**: Measurable cost reduction with a full, immutable audit trail.

---

## 🎯 What You Learn

| Concept | Where |
|---|---|
| **PydanticAI** typed agent contracts | `src/agents/` |
| **LangGraph** multi-agent orchestration | `src/graph/` |
| **MCP enterprise servers** (documents, policies, audit DB) | `src/mcp/` |
| **Langfuse** traces, costs, prompt management | `src/observability/` |
| **LangSmith** evaluation and dataset management | `src/observability/` |
| **AWS AgentCore** compliance runtime | `src/runtime/aws_agentcore.py` |
| **Google Vertex AI** Agent Engine | `src/runtime/vertex_ai.py` |
| **Audit trail** — immutable record of every decision | `src/audit/` |
| **Cost measurement** — before vs after automation | `src/reporting/` |
| **Human-in-the-loop** — interrupt on high-risk items | `src/graph/nodes.py` |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ENTERPRISE COMPLIANCE REVIEW SYSTEM                        │
│                       LangGraph Orchestration                               │
├──────────────┬──────────────────────────────────┬───────────────────────────┤
│  MCP SERVERS │     MULTI-AGENT WORKFLOW         │  OBSERVABILITY            │
│              │                                  │                           │
│ ┌──────────┐ │  ┌──────────┐  ┌──────────────┐  │ ┌─────────────────────┐   │
│ │Document  │ │  │ Intake   │─►│  Risk        │  │ │ Langfuse / LangSmith│   │
│ │Store MCP │◄├──│ Agent    │  │  Analysis    │  │ │ ─ Traces per run    │   │
│ └──────────┘ │  │(validate)│  │  Agent       │  │ │ ─ Cost per document │   │
│              │  └──────────┘  │(PydanticAI)  │  │ │ ─ Prompt versions   │   │
│ ┌──────────┐ │                └──────┬───────┘  │ │ ─ SLA dashboard     │   │
│ │Regulatory│ │               ┌──────▼───────┐   │ └─────────────────────┘   │
│ │Policy MCP│◄├───────────────│  Policy      │   │                           │
│ └──────────┘ │               │  Check Agent │   │  PYDANTIC AI CONTRACTS    │
│              │               │(PydanticAI)  │   │                           │
│ ┌──────────┐ │               └──────┬───────┘   │ ┌─────────────────────┐   │
│ │Audit DB  │ │               ┌──────▼───────┐   │ │ RiskAssessment      │   │
│ │MCP       │◄├───────────────│  ⏸ INTERRUPT │   │ │ PolicyViolation     │   │
│ └──────────┘ │               │  Human HITL  │   │ │ AuditEntry          │   │
│              │               └──────┬───────┘   │ │ ComplianceReport    │   │
│              │               ┌──────▼───────┐   │ └─────────────────────┘   │
│              │               │  Audit       │   │                           │
│              │               │  Logger      │   │  MANAGED RUNTIME          │
│              │               └──────┬───────┘   │                           │
│              │               ┌──────▼───────┐   │ ┌─────────────────────┐   │
│              │               │  Report      │   │ │ AWS AgentCore       │   │
│              │               │  Generator   │   │ │ (compliance/audit)  │   │
│              │               └──────────────┘   │ │ ─OR─                │   │
│              │                                  │ │ Google Vertex AI    │   │
│              │                                  │ │ (analytics)         │   │
└──────────────┴──────────────────────────────────┴─└─────────────────────┘───┘

Conditional edges:
  risk=LOW     → skip HITL → auto-approve → audit → report
  risk=MEDIUM  → policy check → auto-approve if compliant → audit → report
  risk=HIGH    → INTERRUPT → human review → audit → report
  risk=CRITICAL→ INTERRUPT → human review → legal escalation
```

---

## 📁 Project Structure

```
project23_enterprise_architect/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   └── src/
│       ├── agents/         PydanticAI agent stubs
│       ├── graph/          LangGraph state + node stubs
│       ├── mcp/            MCP server stubs
│       └── observability/  Langfuse/LangSmith stubs
└── solution/
    ├── .env.example
    └── src/
        ├── config.py
        ├── main.py
        ├── agents/
        │   ├── risk_agent.py        PydanticAI risk assessor
        │   └── policy_agent.py      PydanticAI policy checker
        ├── graph/
        │   ├── state.py             ComplianceState TypedDict
        │   ├── nodes.py             All workflow nodes
        │   └── graph.py             StateGraph builder
        ├── mcp/
        │   ├── document_server.py   MCP document store
        │   ├── policy_server.py     MCP policy database
        │   └── audit_server.py      MCP audit log server
        ├── observability/
        │   ├── langfuse_tracer.py   Langfuse integration
        │   └── langsmith_tracer.py  LangSmith integration
        ├── runtime/
        │   ├── aws_agentcore.py     AWS AgentCore deployment
        │   └── vertex_ai.py         Vertex AI Agent Engine
        ├── audit/
        │   └── audit_trail.py       Immutable audit record
        └── reporting/
            └── cost_report.py       Cost savings measurement
```

---

## 🚀 Quick Start

```bash
cd project23_enterprise_architect/solution
cp .env.example .env    # fill in your keys
pip install -r ../starter/requirements.txt

# Start MCP servers (in separate terminals)
python -m src.mcp.document_server
python -m src.mcp.policy_server
python -m src.mcp.audit_server

# Run a compliance review
python -m src.main review --doc-id DOC-2024-001 --type gdpr

# Start the FastAPI service
python -m src.main api
```

---

## 🏆 Milestones

| Milestone | Deliverable | Success Criterion |
|---|---|---|
| **M1** | PydanticAI contracts compiling | All models validated, type errors caught at dev time |
| **M2** | LangGraph workflow running end-to-end | All 5 nodes execute, state flows correctly |
| **M3** | MCP servers connected | Documents retrieved, audit entries written via MCP |
| **M4** | HITL working | Graph pauses on HIGH risk, resumes on human input |
| **M5** | Langfuse traces live | Every run visible in dashboard with cost breakdown |
| **M6** | Cost measurement baseline | Before/after comparison showing ≥ 40% cost reduction |
| **Q1 Goal** | Full compliance review pipeline in production | Audit trail, cost savings report, SLA metrics |

---

## 📊 Expected Business Outcomes

| Metric | Manual Process | Automated (Target) |
|---|---|---|
| Time per document | 4–6 hours | 8–12 minutes |
| Cost per document | $150–250 (analyst time) | $2–5 (LLM + compute) |
| Audit trail completeness | Partial (emails, notes) | 100% immutable log |
| Throughput | 10–20 docs/day | 500+ docs/day |
| Error rate | 3–8% (human error) | < 0.5% (validation gates) |
