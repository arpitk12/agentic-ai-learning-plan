# Week 11 — System Integration

## Topics
1. Microservices architecture for agent systems
2. MCP (Model Context Protocol) for tool standardization
3. Database design for agent state, memories, run history
4. CI/CD pipelines with agent regression tests

## Key Concepts

### MCP (Model Context Protocol)
MCP standardizes how agents connect to tools and data sources.
Instead of hardcoding tools, agents discover them dynamically.

```
Agent ←→ MCP Client ←→ MCP Server ←→ Tool/Data Source
```

Benefits:
- Tool reuse across agents
- Versioned tool interfaces
- Sandboxed execution
- Auth handled at server level

### Database Schema for Agents
```sql
-- Agent runs
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    status VARCHAR(20),  -- pending/running/done/failed
    model VARCHAR(50),
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_cost_usd DECIMAL(10,6),
    result TEXT
);

-- Tool calls within a run
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(id),
    step_number INT,
    tool_name VARCHAR(100),
    inputs JSONB,
    result TEXT,
    duration_ms INT,
    called_at TIMESTAMP
);

-- Long-term agent memory
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY,
    user_id VARCHAR(100),
    memory_type VARCHAR(50),  -- episodic/semantic/preference
    content TEXT,
    embedding VECTOR(1536),   -- pgvector
    created_at TIMESTAMP
);
```

### CI/CD for Agents
```yaml
# .github/workflows/agent_ci.yml
name: Agent Eval CI
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/eval/ -v --tb=short
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: python run_evals.py --fail-below 0.85
```

## Exercises
- `ex1_mcp_server.py` — expose tools as MCP server
- `ex2_db_schema.py` — set up Postgres schema, log agent runs
- `ex3_ci_pipeline.yml` — GitHub Actions eval workflow

## Checklist
- [ ] Tools migrated to MCP server
- [ ] Postgres schema storing all runs and tool calls
- [ ] CI pipeline runs evals on every PR
- [ ] PR blocked if eval pass rate drops below threshold
