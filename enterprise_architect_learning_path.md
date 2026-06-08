# Enterprise Architect Learning Path

> **Stack**: LangGraph + LangChain · PydanticAI · MCP · AWS AgentCore / Google Vertex AI · Langfuse / LangSmith  
> **Q1 Goal**: Multi-agent compliance workflow · measurable cost reduction · full audit trail

Related: [`projects/project23_enterprise_architect/`](projects/project23_enterprise_architect/) · [`framework_selection_guide.md`](framework_selection_guide.md) · [`token_optimization_guide.md`](token_optimization_guide.md)

---

## Table of Contents

1. [Stack Overview](#1-stack-overview)
2. [PydanticAI — Typed Agent Contracts](#2-pydanticai--typed-agent-contracts)
3. [LangGraph Advanced Patterns](#3-langgraph-advanced-patterns)
4. [MCP — Enterprise-Grade Connectivity](#4-mcp--enterprise-grade-connectivity)
5. [AWS AgentCore](#5-aws-agentcore)
6. [Google Vertex AI Agent Engine](#6-google-vertex-ai-agent-engine)
7. [Langfuse — Observability](#7-langfuse--observability)
8. [LangSmith — Evaluation](#8-langsmith--evaluation)
9. [Compliance and Audit Trail Patterns](#9-compliance-and-audit-trail-patterns)
10. [Q1 Roadmap — 12 Weeks to Production](#10-q1-roadmap--12-weeks-to-production)
11. [References and Resources](#11-references-and-resources)

---

## 1. Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   ENTERPRISE ARCHITECT STACK                    │
├──────────────────┬────────────────────────┬─────────────────────┤
│   CONTRACTS      │    ORCHESTRATION       │   CONNECTIVITY      │
│                  │                        │                     │
│  PydanticAI      │  LangGraph             │  MCP Servers        │
│  ─ Typed outputs │  ─ StateGraph          │  ─ Document Store   │
│  ─ Dep injection │  ─ Interrupt/resume    │  ─ Policy DB        │
│  ─ ModelRetry    │  ─ Audit reducer       │  ─ Audit Log        │
│  ─ Tool calling  │  ─ Conditional edges   │  ─ External APIs    │
│                  │                        │                     │
│  LangChain       │  SqliteSaver /         │  Enterprise tools   │
│  ─ LCEL chains   │  PostgresSaver         │  SharePoint / S3    │
│  ─ Tool wrappers │  ─ Persistent state    │  Databases          │
├──────────────────┼────────────────────────┼─────────────────────┤
│   OBSERVABILITY  │    RUNTIME             │   COMPLIANCE        │
│                  │                        │                     │
│  Langfuse        │  AWS AgentCore         │  Audit trail        │
│  ─ Traces        │  ─ IAM auth            │  ─ Hash-chained     │
│  ─ Cost/token    │  ─ VPC isolation       │  ─ Immutable log    │
│  ─ Prompt mgmt   │  ─ CloudTrail logs     │  ─ SOC2 / HIPAA     │
│                  │  ─ Compliance ready    │                     │
│  LangSmith       │                        │  Cost reporting     │
│  ─ Evaluation    │  Google Vertex AI      │  ─ Before/after     │
│  ─ Datasets      │  ─ Managed runtime     │  ─ ROI calculation  │
│  ─ Experiments   │  ─ BigQuery analytics  │  ─ SLA metrics      │
└──────────────────┴────────────────────────┴─────────────────────┘
```

**Why this combination?**

| Need | Solution | Why |
|---|---|---|
| Type-safe agent I/O | PydanticAI | Catch schema violations at dev time, not in prod at 3am |
| Stateful workflow with HITL | LangGraph | Native `interrupt()`, SqliteSaver, conditional edges |
| Enterprise data connectivity | MCP | Decoupled, versionable, auditable data access layer |
| Cost tracking per document | Langfuse | Open-source, self-hostable, token/cost per trace |
| Regression testing | LangSmith | Golden datasets, experiment comparison, CI integration |
| Compliance runtime | AWS AgentCore | CloudTrail, VPC, IAM — required for SOC2/HIPAA |
| Analytics at scale | Vertex AI | BigQuery integration, managed GPU, global deployment |

---

## 2. PydanticAI — Typed Agent Contracts

### What is PydanticAI?

PydanticAI is a Python framework from the Pydantic team that brings type safety to AI agents. Instead of parsing raw JSON from LLM responses, you define a Pydantic model and PydanticAI enforces it — with automatic retries if the model returns the wrong schema.

### Core Pattern

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from typing import Literal

# 1. Define the contract
class RiskAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high", "critical"]
    risk_factors: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

# 2. Create typed agent
risk_agent = Agent(
    model="openai:gpt-4o-mini",
    result_type=RiskAssessment,       # ← enforces the contract
    system_prompt="You are a risk analyst...",
)

# 3. Validate + auto-retry
@risk_agent.result_validator
async def validate(ctx: RunContext, result: RiskAssessment) -> RiskAssessment:
    if result.confidence < 0.7 and result.risk_level == "critical":
        raise ModelRetry("Provide specific citations for CRITICAL assessments")
    return result

# 4. Run — result.data is guaranteed RiskAssessment
result = await risk_agent.run("Review this contract...")
assessment: RiskAssessment = result.data   # fully typed
```

### Dependency Injection

```python
from dataclasses import dataclass

@dataclass
class Deps:
    policies: dict[str, str]
    user_role: str

agent = Agent(model="openai:gpt-4o-mini", result_type=Output, deps_type=Deps)

@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[Deps]) -> str:
    return f"User role: {ctx.deps.user_role}. Enforce {len(ctx.deps.policies)} policies."

@agent.tool
async def get_policy(ctx: RunContext[Deps], policy_id: str) -> str:
    return ctx.deps.policies.get(policy_id, "Not found")

result = await agent.run("Check this doc", deps=Deps(policies={...}, user_role="analyst"))
```

### Key Concepts

| Concept | API | Use case |
|---|---|---|
| Typed output | `Agent(result_type=MyModel)` | Enforce schema, get IDE completion |
| Retry on bad output | `@agent.result_validator` + `ModelRetry` | Self-healing agents |
| Tool calling | `@agent.tool` | Inject functions the LLM can call |
| Dynamic system prompts | `@agent.system_prompt` | Context-dependent instructions |
| Dependency injection | `deps_type=Deps` | Pass runtime config without globals |
| Streaming | `async for chunk in agent.run_stream(...)` | Real-time token output |

### Learning Resources

| Resource | Link |
|---|---|
| PydanticAI docs | [ai.pydantic.dev](https://ai.pydantic.dev) |
| PydanticAI GitHub | [github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) |
| PydanticAI examples | [ai.pydantic.dev/examples](https://ai.pydantic.dev/examples/) |
| Getting started tutorial | [ai.pydantic.dev/install](https://ai.pydantic.dev/install/) |

```bash
pip install pydantic-ai
```

---

## 3. LangGraph Advanced Patterns

### Parallel Nodes (Fan-Out / Fan-In)

Run multiple analysis agents simultaneously and merge results:

```python
from langgraph.graph import StateGraph, START, END
import asyncio

# Fan-out: risk and policy checks in parallel
async def parallel_analysis(state):
    risk_task = asyncio.create_task(assess_risk(state["content"]))
    policy_task = asyncio.create_task(check_policies(state["content"]))
    risk, policy = await asyncio.gather(risk_task, policy_task)
    return {"risk": risk, "policy": policy}
```

### Subgraphs — Modular Workflow Composition

```python
# Build a reusable analysis subgraph
def build_analysis_subgraph():
    builder = StateGraph(AnalysisState)
    builder.add_node("risk", risk_node)
    builder.add_node("policy", policy_node)
    builder.add_edge(START, "risk")
    builder.add_edge("risk", "policy")
    builder.add_edge("policy", END)
    return builder.compile()

analysis_graph = build_analysis_subgraph()

# Embed as a node in the parent graph
main_builder.add_node("analysis", analysis_graph)  # subgraph as node
```

### PostgresSaver — Production Checkpointing

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncpg

async def get_production_checkpointer():
    pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()   # creates checkpoint tables
    return checkpointer
```

### Multi-agent Supervisor Pattern

```python
# Supervisor decides which specialist agent to call next
supervisor_agent = Agent(
    model="openai:gpt-4o",
    result_type=SupervisorDecision,
    system_prompt="You coordinate: risk_analyst, policy_checker, legal_reviewer. "
                  "Decide who should act next based on current state.",
)

async def supervisor_node(state):
    decision = await supervisor_agent.run(str(state))
    return {"next_agent": decision.data.agent, "instructions": decision.data.instructions}
```

### Learning Resources

| Resource | Link |
|---|---|
| LangGraph how-to guides | [langchain-ai.github.io/langgraph/how-tos](https://langchain-ai.github.io/langgraph/how-tos/) |
| LangGraph Academy (free) | [academy.langchain.com](https://academy.langchain.com) |
| Multi-agent with LangGraph | [langchain-ai.github.io/langgraph/tutorials/multi_agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/) |
| LangGraph conceptual guide | [langchain-ai.github.io/langgraph/concepts](https://langchain-ai.github.io/langgraph/concepts/) |
| Subgraphs how-to | [langchain-ai.github.io/langgraph/how-tos/subgraph](https://langchain-ai.github.io/langgraph/how-tos/subgraph/) |

---

## 4. MCP — Enterprise-Grade Connectivity

### What is MCP?

Model Context Protocol (MCP) is an open standard (Anthropic, 2024) for connecting AI agents to data sources and tools. Think of it as USB-C for AI — a single protocol for document stores, databases, APIs, and file systems.

### Why Enterprise MCP vs Direct Access?

| Approach | Problem | MCP Solution |
|---|---|---|
| Direct DB connection in agent | Hard-coded credentials, no audit | MCP server with IAM + access logging |
| REST API calls in agent code | Version coupling, no centralized control | MCP server with versioning + gateway |
| File system access in agent | No access control, no audit trail | MCP resource server with permissions |

### Enterprise MCP Server Patterns

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enterprise-document-store", version="2.1.0")

# Tools — actions the agent can take
@mcp.tool()
async def get_document(doc_id: str, classification: str = "internal") -> dict:
    """Retrieve document. Access logged automatically."""
    # MCP handles: auth, rate limiting, audit logging
    return {"content": ..., "metadata": ..., "retrieved_at": ...}

# Resources — data the agent can read
@mcp.resource("documents://schema")
async def document_schema() -> str:
    return "JSON Schema for all document types"

# Prompts — reusable prompt templates
@mcp.prompt()
async def compliance_review_prompt(doc_type: str) -> list[dict]:
    return [{"role": "user", "content": f"Review this {doc_type} for compliance..."}]
```

### Connecting LangGraph to MCP

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def fetch_via_mcp(doc_id: str) -> str:
    """Call MCP document server from within a LangGraph node."""
    async with stdio_client(
        {"command": "python", "args": ["-m", "src.mcp.document_server"]}
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_document", {"doc_id": doc_id})
            return result.content[0].text
```

### Learning Resources

| Resource | Link |
|---|---|
| MCP specification | [modelcontextprotocol.io](https://modelcontextprotocol.io) |
| FastMCP (Python) | [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp) |
| MCP Python SDK | [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) |
| Enterprise MCP patterns | [modelcontextprotocol.io/docs/concepts/architecture](https://modelcontextprotocol.io/docs/concepts/architecture) |
| LangChain MCP integration | [python.langchain.com/docs/integrations/tools/mcp](https://python.langchain.com/docs/integrations/tools/mcp/) |

```bash
pip install mcp fastmcp
```

---

## 5. AWS AgentCore

### What is AWS AgentCore?

Amazon Bedrock AgentCore (launched 2025) is a managed runtime for AI agents that provides:
- **Managed execution environment** — no infra to provision
- **IAM-based authentication** — fine-grained access control
- **VPC isolation** — agents run in your private network
- **CloudTrail integration** — every agent action is audit-logged
- **Built-in compliance** — SOC2, HIPAA, PCI-DSS ready
- **Memory management** — conversation history stored and retrieved automatically

### When to Choose AWS AgentCore

```
Use AgentCore when:
  ✅ Workload requires SOC2/HIPAA/PCI-DSS compliance
  ✅ Already deep in AWS ecosystem (IAM, VPC, CloudTrail)
  ✅ Need VPC isolation for sensitive data
  ✅ Want managed scaling without Kubernetes

Use Vertex AI instead when:
  ✅ Need BigQuery integration for analytics
  ✅ Using Google Workspace data sources
  ✅ Want Gemini as primary model
  ✅ Need global multi-region deployment
```

### Deployment Pattern

```python
import boto3, json

# 1. Package your LangGraph agent as a Lambda or container
# 2. Register with AgentCore
client = boto3.client("bedrock-agentcore", region_name="us-east-1")

# 3. Invoke via AgentCore (handles auth, logging, scaling)
response = client.invoke_agent(
    agentId="agent-xxxxxxxxxxxx",
    sessionId="session-001",
    inputText=json.dumps({"document_id": "DOC-001"}),
)

# 4. CloudTrail automatically logs every invocation
# 5. VPC endpoint keeps data in your network
```

### Learning Resources

| Resource | Link |
|---|---|
| AWS AgentCore docs | [docs.aws.amazon.com/bedrock/latest/userguide/agents-core.html](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-core.html) |
| AWS re:Invent AgentCore talk | [youtube.com — search "AgentCore re:Invent 2025"](https://www.youtube.com/results?search_query=AWS+AgentCore+2025) |
| Bedrock compliance | [aws.amazon.com/compliance/services-in-scope](https://aws.amazon.com/compliance/services-in-scope/) |
| LangGraph on AWS | [python.langchain.com/docs/integrations/platforms/aws](https://python.langchain.com/docs/integrations/platforms/aws/) |

```bash
pip install boto3
aws configure   # set access key, secret, region
```

---

## 6. Google Vertex AI Agent Engine

### What is Vertex AI Agent Engine?

Vertex AI Agent Engine (formerly Vertex AI Extensions + Reasoning Engine) is Google's managed runtime for LangChain and LangGraph agents:
- **One-command deployment** — `ReasoningEngine.create(agent, requirements=[...])`
- **Automatic scaling** — handles 1 to 1M+ invocations
- **BigQuery integration** — agent outputs flow directly to BQ for analytics
- **Gemini models** — native integration with Gemini 1.5 Pro / Flash
- **Vertex AI Pipelines** — batch processing of thousands of documents

### Deployment

```python
import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine

vertexai.init(project="my-project", location="us-central1")

# Your LangGraph app wrapped in a class
class ComplianceApp:
    def query(self, document_id: str, document_type: str) -> dict:
        # LangGraph invocation here
        ...

# Deploy to Vertex AI — one call, managed forever
engine = ReasoningEngine.create(
    ComplianceApp(),
    requirements=["langgraph>=0.2", "pydantic-ai>=0.1", "langfuse>=3.0"],
    display_name="compliance-review-v1",
)

# Invoke from anywhere
result = engine.query(document_id="DOC-001", document_type="contract")

# Scale: process thousands of documents in batch
import asyncio
results = await asyncio.gather(*[
    engine.aquery(document_id=doc_id, document_type="contract")
    for doc_id in doc_ids
])
```

### Learning Resources

| Resource | Link |
|---|---|
| Vertex AI Agent Engine | [cloud.google.com/vertex-ai/generative-ai/docs/agent-engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview) |
| LangGraph on Vertex AI | [cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/langgraph](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/langgraph) |
| Vertex AI pricing | [cloud.google.com/vertex-ai/pricing](https://cloud.google.com/vertex-ai/pricing) |
| Gemini API docs | [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs) |

```bash
pip install google-cloud-aiplatform vertexai
gcloud auth application-default login
```

---

## 7. Langfuse — Observability

### What is Langfuse?

Langfuse is an open-source LLM observability platform. Unlike proprietary tools, you can self-host it (Docker compose) for full data privacy — critical for compliance workloads.

**Key features:**
- **Traces** — full call tree for every agent run (which node called which LLM, with latency)
- **Cost tracking** — token count and USD cost per trace, per user, per model
- **Prompt management** — versioned prompt templates with A/B testing
- **Evaluations** — built-in and custom LLM-as-judge scoring
- **Datasets** — curate test cases for regression testing

### Integration Patterns

```python
# Pattern 1: LangChain callback (automatic)
from langfuse.callback import CallbackHandler
handler = CallbackHandler(session_id="session-001")
chain.invoke(input, config={"callbacks": [handler]})

# Pattern 2: Decorator (PydanticAI / any async function)
from langfuse.decorators import observe
@observe(name="risk-analysis")
async def run_risk(document: str) -> dict:
    result = await risk_agent.run(document)
    return result.data.model_dump()

# Pattern 3: Manual tracing (full control)
from langfuse import Langfuse
langfuse = Langfuse()
trace = langfuse.trace(name="compliance-review", session_id="session-001")
span = trace.span(name="risk-analysis", input={"document": doc})
# ... run analysis ...
span.end(output={"risk_level": "high"})
trace.update(output={"status": "completed"})
```

### Cost Dashboard Setup

```python
# Query Langfuse for cost by document type
def compliance_cost_report(days: int = 30) -> None:
    client = Langfuse()
    traces = client.get_traces(tags=["compliance"], limit=1000)

    total_cost = 0
    for trace in traces.data:
        for obs in (trace.observations or []):
            total_cost += obs.calculated_total_cost or 0

    print(f"Total LLM cost (last {days}d): ${total_cost:.2f}")
    print(f"Cost per document: ${total_cost/len(traces.data):.4f}")
```

### Self-Hosting Langfuse

```yaml
# docker-compose.yml
version: "3"
services:
  langfuse:
    image: langfuse/langfuse:3
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/langfuse
      NEXTAUTH_SECRET: your-secret-here
      SALT: your-salt-here
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: langfuse
```

### Learning Resources

| Resource | Link |
|---|---|
| Langfuse docs | [langfuse.com/docs](https://langfuse.com/docs) |
| Langfuse Python SDK | [langfuse.com/docs/sdk/python](https://langfuse.com/docs/sdk/python) |
| LangChain integration | [langfuse.com/docs/integrations/langchain](https://langfuse.com/docs/integrations/langchain) |
| Self-hosting guide | [langfuse.com/docs/deployment/self-host](https://langfuse.com/docs/deployment/self-host) |
| Langfuse GitHub | [github.com/langfuse/langfuse](https://github.com/langfuse/langfuse) |
| Cost tracking docs | [langfuse.com/docs/model-usage-and-cost](https://langfuse.com/docs/model-usage-and-cost) |

```bash
pip install langfuse
```

---

## 8. LangSmith — Evaluation

### When to Use LangSmith vs Langfuse

| Feature | LangSmith | Langfuse |
|---|---|---|
| Primary strength | Evaluation + datasets | Observability + cost |
| LangChain integration | Native (same company) | Via callback |
| Dataset management | ✅ Best-in-class | ✅ Supported |
| Self-hosting | ❌ SaaS only | ✅ Docker / K8s |
| A/B prompt testing | ✅ | ✅ |
| LLM-as-judge evaluators | ✅ | ✅ |
| Cost tracking | ⚠️ Basic | ✅ Detailed |

**Recommended combination**: Langfuse for prod cost/trace monitoring + LangSmith for evaluation datasets and CI evaluation gates.

### Setting Up Evaluation Pipeline

```python
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

client = Client()

# 1. Create golden dataset
client.create_dataset("compliance-golden-v1")
client.create_examples(
    inputs=[{"document": doc, "doc_type": "contract"} for doc in test_docs],
    outputs=[{"expected_risk": risk} for risk in expected_risks],
    dataset_name="compliance-golden-v1",
)

# 2. Define evaluator
def risk_level_accuracy(run, example) -> dict:
    predicted = run.outputs.get("risk_level")
    expected = example.outputs.get("expected_risk")
    return {"key": "risk_accuracy", "score": 1 if predicted == expected else 0}

# 3. Run evaluation (CI-friendly)
results = evaluate(
    traced_risk_analysis,        # @traceable function
    data="compliance-golden-v1",
    evaluators=[risk_level_accuracy],
    experiment_prefix="risk-agent-",
)
print(f"Risk accuracy: {results.to_pandas()['risk_accuracy'].mean():.0%}")
```

### Learning Resources

| Resource | Link |
|---|---|
| LangSmith docs | [docs.smith.langchain.com](https://docs.smith.langchain.com) |
| Evaluation tutorial | [docs.smith.langchain.com/evaluation](https://docs.smith.langchain.com/evaluation) |
| Dataset management | [docs.smith.langchain.com/evaluation/faq/datasets](https://docs.smith.langchain.com/evaluation/faq/datasets) |
| DeepLearning.AI: LangSmith course | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/langsmith/) |

---

## 9. Compliance and Audit Trail Patterns

### Immutable Audit Trail — Hash-Chained JSONL

Every entry hashes the previous entry's hash — if any entry is tampered with, the chain breaks.

```python
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path("data/audit_trail.jsonl")

def write_audit_entry(doc_id: str, step: str, actor: str, details: dict) -> str:
    """Append tamper-evident audit entry. Returns entry hash."""
    prev_hash = _get_last_hash()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id, "step": step, "actor": actor, "details": details,
        "prev_hash": prev_hash,
    }
    entry["hash"] = hashlib.sha256(
        (prev_hash + json.dumps(entry, sort_keys=True)).encode()
    ).hexdigest()
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]

def verify_audit_trail() -> bool:
    """Verify the hash chain is unbroken."""
    entries = [json.loads(l) for l in AUDIT_FILE.read_text().splitlines() if l]
    for i, entry in enumerate(entries[1:], 1):
        expected_prev = entries[i-1]["hash"]
        if entry["prev_hash"] != expected_prev:
            print(f"❌ Chain broken at entry {i}: {entry['step']}")
            return False
    print(f"✅ Audit trail verified: {len(entries)} entries, chain intact")
    return True
```

### Compliance-Friendly LLM Patterns

```python
# 1. Never log raw document content (PII risk)
logger.info("Processing document", doc_id=doc_id, doc_type=doc_type)  # ✅
logger.info("Processing document", content=document_content)           # ❌

# 2. Always record model + version for reproducibility
audit_entry["model"] = cfg.model
audit_entry["model_version"] = litellm.get_model_info(cfg.model).get("version")

# 3. Record prompt hash, not prompt text (for prompt injection audit)
audit_entry["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()

# 4. Store confidence scores for human review decisions
audit_entry["confidence"] = assessment.confidence
audit_entry["human_review_threshold"] = 0.85

# 5. Record data lineage
audit_entry["source_doc_hash"] = hashlib.sha256(document_content.encode()).hexdigest()
```

### Regulatory Frameworks Reference

| Framework | Scope | Key Requirements for AI |
|---|---|---|
| **GDPR** | EU personal data | Data minimization, right to explanation, DPA for processors |
| **SOX** | US financial reporting | 7-year retention, change management, access control |
| **HIPAA** | US healthcare data | PHI encryption, access logging, BAA for vendors |
| **PCI-DSS** | Payment card data | Encryption, network segmentation, audit logging |
| **ISO 27001** | Information security | Risk management, asset inventory, incident response |
| **SOC 2** | Service organizations | Security, availability, confidentiality, privacy |

---

## 10. Q1 Roadmap — 12 Weeks to Production

### Week 1–2: Foundations
- [ ] Complete Project 23 Phase 1 (PydanticAI contracts)
- [ ] Define your compliance document taxonomy (types, risk levels)
- [ ] Set up Langfuse locally (Docker compose)
- [ ] Baseline: measure current manual review time and cost

### Week 3–4: Workflow
- [ ] Complete Project 23 Phase 2 (LangGraph workflow)
- [ ] Test HITL interrupt/resume with realistic documents
- [ ] Implement audit trail and verify hash chain
- [ ] Create golden dataset of 50 documents with known outcomes

### Week 5–6: Connectivity
- [ ] Complete Project 23 Phase 3 (MCP servers)
- [ ] Connect to real document store (SharePoint/S3)
- [ ] Build policy database MCP server
- [ ] End-to-end test with real documents from your organization

### Week 7–8: Observability
- [ ] Complete Project 23 Phase 4 (Langfuse + LangSmith)
- [ ] Set up cost-per-document dashboard in Langfuse
- [ ] Create LangSmith evaluation pipeline with 50-doc golden dataset
- [ ] Define SLA: target < 15 minutes per document

### Week 9–10: Runtime
- [ ] Choose runtime: AWS AgentCore (compliance) or Vertex AI (analytics)
- [ ] Deploy to staging environment
- [ ] Load test: 100 documents/hour throughput
- [ ] Security review: prompt injection, PII handling, access control

### Week 11–12: Production Rollout
- [ ] Parallel run: automated + manual for 2 weeks
- [ ] Compare accuracy: automated vs human baseline
- [ ] Measure cost savings (target: ≥ 60% reduction)
- [ ] Go-live with 10% of document volume, scale to 100%
- [ ] Present Q1 report: throughput, cost, accuracy, audit completeness

### Success Metrics (Q1)

| Metric | Target | How to Measure |
|---|---|---|
| Cost per document | < $5 (vs $425 manual) | Langfuse cost tracking |
| Processing time | < 15 minutes (vs 5 hours) | Langfuse latency |
| Audit trail completeness | 100% | Hash chain verification |
| Accuracy vs human baseline | ≥ 95% on low/medium risk | LangSmith evaluation |
| Human review rate | ≤ 20% of documents | LangGraph route_by_risk stats |
| Monthly cost reduction | ≥ 60% | CostComparison.savings_pct |

---

## 11. References and Resources

### 📄 Papers and Standards

| Resource | Relevance |
|---|---|
| **"Agents" by Anthropic** (2024) | Foundation for agentic system design patterns | [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents) |
| **Model Context Protocol spec** (2024) | Full MCP protocol specification | [spec.modelcontextprotocol.io](https://spec.modelcontextprotocol.io) |
| **NIST AI Risk Management Framework** | AI governance for enterprise | [nist.gov/artificial-intelligence/ai-rmf](https://www.nist.gov/artificial-intelligence/ai-rmf) |
| **EU AI Act** (2024) | Regulatory compliance for AI in EU | [eur-lex.europa.eu/AI-Act](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) |

### 📚 Official Documentation

| Tool | Link |
|---|---|
| PydanticAI | [ai.pydantic.dev](https://ai.pydantic.dev) |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| FastMCP | [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp) |
| Langfuse | [langfuse.com/docs](https://langfuse.com/docs) |
| LangSmith | [docs.smith.langchain.com](https://docs.smith.langchain.com) |
| AWS AgentCore | [docs.aws.amazon.com/bedrock/agentcore](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-core.html) |
| Vertex AI Agent Engine | [cloud.google.com/vertex-ai/agent-engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview) |

### 🎓 Courses

| Course | Link |
|---|---|
| DeepLearning.AI: AI Agents in LangGraph | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) |
| LangGraph Academy | [academy.langchain.com](https://academy.langchain.com) |
| AWS: Building AI Agents on Bedrock | [explore.skillbuilder.aws](https://explore.skillbuilder.aws) |
| Google: Generative AI on Vertex AI | [cloudskillsboost.google](https://cloudskillsboost.google/paths/183) |
| Coursera: AI for Compliance | [coursera.org](https://www.coursera.org/search?query=AI+compliance) |

### 📖 Books

| Book | Relevance |
|---|---|
| **"AI Engineering"** — Chip Huyen (2025) | Production AI systems, cost optimization |
| **"Building LLM Apps"** — Valentina Alto (2024) | LangChain + LangGraph practical guide |
| **"Designing Data-Intensive Applications"** — Kleppmann | Audit trail, data pipeline patterns |

### 🔗 Community

| Community | Link |
|---|---|
| LangChain Discord (LangGraph channel) | [discord.gg/langchain](https://discord.gg/langchain) |
| Langfuse Discord | [discord.gg/langfuse](https://discord.gg/langfuse) |
| PydanticAI GitHub Discussions | [github.com/pydantic/pydantic-ai/discussions](https://github.com/pydantic/pydantic-ai/discussions) |
| r/MachineLearning (enterprise AI posts) | [reddit.com/r/MachineLearning](https://reddit.com/r/MachineLearning) |

---

*Also see:*
- *[`projects/project23_enterprise_architect/`](projects/project23_enterprise_architect/) — Full implementation*
- *[`framework_selection_guide.md`](framework_selection_guide.md) — When to use each framework*
- *[`token_optimization_guide.md`](token_optimization_guide.md) — Cut LLM costs 70–90%*
- *[`guide/07_cost_optimization.md`](guide/07_cost_optimization.md) — Cost optimization deep dive*
