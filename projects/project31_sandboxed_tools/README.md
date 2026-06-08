# Project 31 — Sandboxed Tool Execution (E2B + Docker)

> **Stack**: E2B · Docker SDK · asyncio · LangGraph  
> **Phase 7 — Advanced Production** | Priority: P2 🟡

---

## What You'll Build

A safe code execution environment for agents — so that when your LLM generates Python to analyze data or run calculations, it runs in a completely isolated sandbox rather than on your production server.

**Without sandboxing** (dangerous):
```python
# Agent-generated code runs in your process
exec(agent_generated_code)   # can delete files, exfiltrate secrets, install malware
```

**With sandboxing** (safe):
```python
sandbox = Sandbox()
result = sandbox.run_code(agent_generated_code, timeout=30)
# Runs in isolated container, no access to your filesystem or network
```

---

## Milestones

### Milestone 1 — E2B Basic Execution
Create an E2B sandbox, run Python code, capture stdout/stderr. Test: numpy calculations, pandas analysis, matplotlib chart generation. Verify complete isolation.

### Milestone 2 — Agent + E2B Integration
Build a data analyst agent (LangGraph) where the code generation tool uses E2B instead of `exec()`. Test with: CSV analysis, SQL queries, statistical calculations.

### Milestone 3 — Docker Sandbox Alternative
For teams without E2B API key: build a Docker-based sandbox. Spin up a Python container, copy code via volume, execute, capture output, destroy container. Add CPU/memory limits.

### Milestone 4 — Timeout + Kill Switches
Both sandboxes: enforce 30-second timeout. Kill stuck processes gracefully. Return `{"error": "timeout", "partial_output": ...}` on timeout.

### Milestone 5 — Reversibility Analysis
Before executing any tool call, classify it as reversible or irreversible. Irreversible: file writes, HTTP POST, database mutations, shell commands with `rm`. Require HITL approval for irreversible actions.

### Milestone 6 — Tool Execution Audit Log
Log every sandbox execution: who called it, what code ran (hash), when, how long, stdout/stderr summary. Store in append-only audit log.

### Milestone 7 — Output Validation
After code executes, validate the output: is it the expected type? Does it contain PII? Is the data within expected bounds? Reject and retry if validation fails.

---

## Setup

```bash
pip install e2b litellm langgraph pydantic python-dotenv docker
# E2B API key: https://e2b.dev (free tier available)
# Docker: install Docker Desktop
```

---

## Expected Output

```
=== Sandboxed Tool Execution ===

E2B Sandbox:
  Code: "import numpy as np; print(np.mean([1,2,3,4,5]))"
  Output: 3.0
  Execution time: 1.2s | Isolated: ✅ | Memory limit: 512MB

Agent data analysis task:
  Query: "What's the average sale by region in sales.csv?"
  Generated code: [agent-generated pandas code]
  Sandbox result: {"East": 42500, "West": 38900, "South": 51200}
  Agent response: "Average sales by region: East $42.5k, West $38.9k, South $51.2k"

Reversibility check:
  Tool: write_file(path="config.json", content="...")
  Classification: IRREVERSIBLE (file write)
  Action: Paused for human approval ⚠️
```
