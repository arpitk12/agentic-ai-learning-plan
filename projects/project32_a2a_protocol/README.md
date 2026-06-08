# Project 32 — A2A Protocol (Agent-to-Agent Communication)

> **Stack**: FastAPI · httpx · JWT · Google A2A Protocol · LangGraph  
> **Phase 7 — Advanced Production** | Priority: P2 🟡

---

## What You'll Build

A network of interoperable agents that can discover and call each other via Google's A2A protocol — your compliance agent can delegate legal review to a separate legal agent, which may itself call a regulatory database agent.

```
                  ┌─────────────────────┐
User Request      │   Compliance Agent   │
(review contract) │ (LangGraph, port 8001)│
                  └─────────┬───────────┘
                            │ A2A task delegation
                            ▼
                  ┌─────────────────────┐
                  │    Legal Agent       │
                  │ (CrewAI, port 8002)  │
                  └─────────┬───────────┘
                            │ A2A task delegation
                            ▼
                  ┌─────────────────────┐
                  │  Regulatory DB Agent │
                  │ (raw litellm, port 8003)│
                  └─────────────────────┘
```

---

## Milestones

### Milestone 1 — Agent Card Endpoint
Create `/.well-known/agent.json` for each agent declaring: name, URL, version, capabilities (streaming, pushNotifications), authentication scheme, skills with input/output modes.

### Milestone 2 — A2A Task Endpoint
Implement `POST /a2a/tasks/send` that accepts a task, runs the agent, and returns the result. Schema: `{id, message: {parts: [{type: "text", text: "..."}]}, skill: "..."}`.

### Milestone 3 — A2A Client
Build a client that: fetches the agent card from a remote agent, sends a task, polls for result (or receives streaming updates). Implement JWT bearer token auth between agents.

### Milestone 4 — Multi-Agent Network
Wire three agents together: compliance → legal → regulatory-db. Build the delegation chain: compliance agent's tool calls legal agent via A2A; legal calls regulatory via A2A. Test end-to-end.

### Milestone 5 — Agent Discovery
Implement a simple agent registry (dict of agent-name → URL). Agents register on startup and query the registry to find collaborators. Add health checking.

### Milestone 6 — Streaming A2A Tasks
Add streaming support: `POST /a2a/tasks/send-streaming` returns Server-Sent Events. Each event: `{type: "progress", message: "..."}` and final `{type: "complete", result: {...}}`.

### Milestone 7 — Cross-Framework Calls
Verify A2A protocol works across frameworks: LangGraph agent calls a CrewAI agent. Since A2A is HTTP-based, any framework works as long as it exposes the endpoints.

---

## Setup

```bash
pip install fastapi uvicorn httpx pyjwt litellm langgraph crewai pydantic python-dotenv
```

---

## Expected Output

```
=== A2A Multi-Agent Network ===

Agents registered:
  compliance-agent  → http://localhost:8001  (skills: compliance_review)
  legal-agent       → http://localhost:8002  (skills: legal_review, contract_risk)
  regulatory-agent  → http://localhost:8003  (skills: regulation_lookup)

Processing: "Review this vendor contract for legal and regulatory compliance"

Compliance agent:
  → Delegating to legal-agent (A2A task send)...
  Legal agent:
    → Delegating to regulatory-agent (A2A task send)...
    Regulatory agent: "GDPR Art. 28 requires: written DPA, data minimization..."
    ← Returning to legal-agent (1.2s)
  Legal agent: "Contract missing DPA exhibit. SOX §404 controls not documented."
  ← Returning to compliance-agent (2.8s)
Compliance agent: "HIGH RISK: Missing DPA (legal), SOX controls not documented (regulatory)"
Total latency: 4.1s (3 A2A hops)
```
