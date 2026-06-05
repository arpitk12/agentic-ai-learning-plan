# Week 11 — System Integration: MCP, Databases & CI/CD

## What This Week Is About
Real agents live in ecosystems — they connect to databases, expose standardized interfaces, integrate with existing software, and deploy through automated pipelines. This week covers the Model Context Protocol (MCP) for agent interoperability, PostgreSQL for production data, and GitHub Actions CI/CD.

---

## 1. Model Context Protocol (MCP) — The Universal Agent Interface

**What it is**: An open protocol (released by Anthropic in 2024) for connecting AI agents to external tools and data sources through a standardized interface. Like USB-C for AI agents.

**Purpose**: Instead of building custom integrations for every data source (Slack, GitHub, Notion, databases, APIs), MCP provides one standard. Any MCP-compatible agent can use any MCP-compatible tool server without custom code.

**Architecture**:
```
Agent (MCP Client)
     │
     │ MCP Protocol (JSON-RPC over stdio or HTTP)
     │
     ▼
MCP Server (e.g., GitHub MCP Server, Postgres MCP Server)
     │
     ▼
Actual Resource (GitHub API, Database, File System)
```

### Building an MCP Server

```python
# mcp_server.py — a simple MCP tool server
import asyncio, json, sys

class MCPServer:
    """Minimal MCP server implementation."""
    
    def __init__(self):
        self.tools = {
            "search_documents": self.search_documents,
            "get_customer": self.get_customer,
            "create_ticket": self.create_ticket,
        }
    
    async def search_documents(self, query: str, limit: int = 5) -> list[dict]:
        """Search internal knowledge base."""
        # In production: query ChromaDB/Qdrant
        return [
            {"id": "doc-1", "content": f"Result for: {query}", "score": 0.95},
        ]
    
    async def get_customer(self, customer_id: str) -> dict:
        """Get customer data from CRM."""
        # In production: query your CRM API or database
        return {"id": customer_id, "name": "Acme Corp", "tier": "enterprise"}
    
    async def create_ticket(self, title: str, description: str, priority: str = "medium") -> dict:
        """Create a support ticket."""
        import uuid
        return {"ticket_id": str(uuid.uuid4()), "status": "created", "title": title}
    
    async def handle_request(self, request: dict) -> dict:
        """Handle an MCP JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {
                "tools": [
                    {"name": name, "description": f"Tool: {name}"}
                    for name in self.tools
                ]
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            if tool_name not in self.tools:
                return {"error": f"Unknown tool: {tool_name}"}
            
            result = await self.tools[tool_name](**tool_args)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        
        return {"error": f"Unknown method: {method}"}
    
    async def run(self):
        """Main loop — read from stdin, write to stdout (MCP stdio transport)."""
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line.strip())
            response = await self.handle_request(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.run())
```

### Connecting as an MCP Client

```python
# mcp_client.py — connect your agent to an MCP server
import subprocess, json, asyncio

class MCPClient:
    def __init__(self, server_command: list[str]):
        self.server_command = server_command
        self.process = None
    
    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE
        )
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        request = json.dumps({
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }) + "\n"
        self.process.stdin.write(request.encode())
        await self.process.stdin.drain()
        
        line = await self.process.stdout.readline()
        response = json.loads(line.decode().strip())
        return response["content"][0]["text"]
    
    async def list_tools(self) -> list:
        request = json.dumps({"method": "tools/list"}) + "\n"
        self.process.stdin.write(request.encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        return json.loads(line.decode())["tools"]

# Usage with Claude Desktop / LangChain / etc.
async def main():
    client = MCPClient(["python", "mcp_server.py"])
    await client.start()
    tools = await client.list_tools()
    result = await client.call_tool("get_customer", {"customer_id": "cust-123"})
    print(result)
```

### Existing MCP Servers (Ready to Use)

```bash
# Install official MCP servers
npx @modelcontextprotocol/server-github     # GitHub integration
npx @modelcontextprotocol/server-postgres   # PostgreSQL
npx @modelcontextprotocol/server-filesystem # File system access
npx @modelcontextprotocol/server-slack      # Slack messaging
```

---

## 2. PostgreSQL for Agent Data

**Why PostgreSQL** (not SQLite) in production:
- Concurrent access from multiple workers
- ACID transactions
- Full-text search with `tsvector`
- JSONB for flexible schema
- Connection pooling via PgBouncer

### Schema Design for Agent Systems

```sql
-- Core schema for a production agent system

-- Users and sessions
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    api_key TEXT UNIQUE NOT NULL DEFAULT encode(gen_random_bytes(32), 'hex'),
    tier TEXT DEFAULT 'free',  -- free, pro, enterprise
    daily_budget_usd DECIMAL(10,4) DEFAULT 1.00,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent conversation sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'  -- agent config, persona, etc.
);

-- Individual agent runs
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    user_id UUID REFERENCES users(id),
    query TEXT NOT NULL,
    result TEXT,
    status TEXT DEFAULT 'pending',  -- pending, running, complete, failed
    model TEXT NOT NULL,
    steps_taken INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6) DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Agent memory / knowledge
CREATE TABLE agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    memory_type TEXT NOT NULL,  -- episodic, semantic, procedural
    content TEXT NOT NULL,
    embedding vector(384),  -- requires pgvector extension
    importance FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Index for vector similarity search
CREATE INDEX ON agent_memories USING ivfflat (embedding vector_cosine_ops);

-- Full-text search index
CREATE INDEX ON agent_memories USING GIN (to_tsvector('english', content));

-- Cost tracking view
CREATE VIEW daily_costs AS
SELECT 
    user_id,
    DATE(started_at) as date,
    COUNT(*) as run_count,
    SUM(cost_usd) as total_cost_usd,
    AVG(steps_taken) as avg_steps
FROM agent_runs
WHERE started_at > NOW() - INTERVAL '30 days'
GROUP BY user_id, DATE(started_at);
```

### Using PostgreSQL from Python

```python
import asyncpg  # pip install asyncpg
import os

async def get_db_pool():
    return await asyncpg.create_pool(
        dsn=os.getenv("DATABASE_URL"),
        min_size=2,
        max_size=10,
        command_timeout=60
    )

# In FastAPI app:
@app.on_event("startup")
async def startup():
    app.state.db = await get_db_pool()

@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    async with app.state.db.acquire() as conn:
        # Record the run
        run_id = await conn.fetchval(
            "INSERT INTO agent_runs (session_id, user_id, query, model, status) VALUES ($1,$2,$3,$4,'running') RETURNING id",
            session_id, user_id, request.query, MODEL
        )
        
        try:
            result = await execute_agent(request.query)
            await conn.execute(
                "UPDATE agent_runs SET result=$1, status='complete', completed_at=NOW() WHERE id=$2",
                result, run_id
            )
            return {"result": result, "run_id": str(run_id)}
        except Exception as e:
            await conn.execute(
                "UPDATE agent_runs SET status='failed', error_message=$1 WHERE id=$2",
                str(e), run_id
            )
            raise
```

---

## 3. GitHub Actions CI/CD

Automated testing and deployment for your agent:

```yaml
# .github/workflows/agent-ci.yml
name: Agent CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"
  MODEL: "gemini/gemini-2.0-flash"

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:alpine
        ports: ["6379:6379"]
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: testpassword
          POSTGRES_DB: agent_test
        ports: ["5432:5432"]
        options: --health-cmd pg_isready
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      
      - name: Install dependencies
        run: pip install -r requirements.txt pytest pytest-asyncio pytest-cov
      
      - name: Run unit tests (no API calls)
        run: pytest tests/unit/ -v --tb=short
      
      - name: Run integration tests (with mock LLM)
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DATABASE_URL: postgresql://postgres:testpassword@localhost/agent_test
          REDIS_URL: redis://localhost:6379/0
        run: pytest tests/integration/ -v --tb=short -x
        
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install ruff mypy
      - run: ruff check .
      - run: mypy . --ignore-missing-imports

  deploy:
    name: Deploy to Production
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t agent-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push agent-api:${{ github.sha }}
      
      - name: Deploy to production
        run: |
          # Update Kubernetes deployment, fly.io, Railway, etc.
          echo "Deploying agent-api:${{ github.sha }}"
```

---

## Tools & Libraries Used This Week

| Tool | Purpose | Install |
|------|---------|---------|
| **MCP SDK** | Model Context Protocol client/server | `pip install mcp` |
| **asyncpg** | Fast async PostgreSQL client | `pip install asyncpg` |
| **pgvector** | Vector similarity in PostgreSQL | Docker extension |
| **GitHub Actions** | CI/CD pipeline | `.github/workflows/*.yml` |
| **Docker** | Container packaging | `docker build` |

---

## Tools Deep Dive — Week 11

### MCP — The USB-C of AI Tool Integration

**The problem before MCP**: Every AI app built its own tool integration protocol. Claude Desktop has one format. OpenAI's function calling has another. LangChain has its tool format. LlamaIndex has yet another. None are compatible.

**MCP's solution**: A standard protocol (JSON-RPC over stdio or HTTP) where:
- **Servers** expose tools, resources, and prompts
- **Clients** (Claude Desktop, agents, IDE plugins) connect to servers
- **Any client can use any server** — like USB-C compatibility

**Why this matters for agents**: Your agent can connect to ANY MCP server — someone else's database server, a file system server, a web search server — without any custom integration code.

```python
# MCP Server — expose your tools to any MCP client
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("my-tools-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_company_db",
            description="Search the company knowledge base. Use for internal documentation, policies, procedures.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_employee_info",
            description="Look up employee information by name or ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Employee name or ID"}
                },
                "required": ["identifier"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_company_db":
        results = await search_db(arguments["query"], arguments.get("max_results", 5))
        return [types.TextContent(type="text", text=json.dumps(results))]
    elif name == "get_employee_info":
        info = await get_employee(arguments["identifier"])
        return [types.TextContent(type="text", text=json.dumps(info))]
    raise ValueError(f"Unknown tool: {name}")

# Run the server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

asyncio.run(main())
```

```python
# MCP Client — your agent using an MCP server's tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],   # your MCP server script
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            
            # Call a tool
            result = await session.call_tool(
                "search_company_db",
                arguments={"query": "vacation policy", "max_results": 3}
            )
            print(result.content[0].text)
```

---

### asyncpg — Why It's Better Than psycopg2 for Async Agents

**psycopg2** is the traditional PostgreSQL driver for Python. It's synchronous — each query blocks the event loop.

**asyncpg** is built for asyncio. Each `await` releases the event loop, allowing other coroutines to run while waiting for the DB response.

```python
import asyncpg
from contextlib import asynccontextmanager
import os

# Connection pool (reuse connections across requests — much faster than reconnecting)
pool = None

async def create_pool():
    global pool
    pool = await asyncpg.create_pool(
        dsn=os.getenv("DATABASE_URL"),
        min_size=2,      # always keep 2 connections ready
        max_size=10,     # maximum 10 concurrent connections
        command_timeout=30,  # query timeout
    )

@asynccontextmanager
async def get_connection():
    async with pool.acquire() as conn:
        yield conn

# Efficient batch operations
async def save_agent_run(run_id: str, user_id: str, query: str, result: str, cost: float):
    async with get_connection() as conn:
        await conn.execute("""
            INSERT INTO agent_runs (run_id, user_id, query, result, cost_usd, completed_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """, run_id, user_id, query, result, cost)  # parameterized — prevents SQL injection

# Batch insert (much faster than individual inserts)
async def save_memories_batch(memories: list[dict]):
    async with get_connection() as conn:
        await conn.executemany("""
            INSERT INTO memories (user_id, content, embedding, category)
            VALUES ($1, $2, $3, $4)
        """, [(m["user_id"], m["content"], m["embedding"], m["category"]) for m in memories])

# Transaction (all-or-nothing)
async def transfer_with_transaction(from_id: str, to_id: str, amount: float):
    async with get_connection() as conn:
        async with conn.transaction():  # rolls back on any exception
            await conn.execute("UPDATE users SET balance = balance - $1 WHERE id = $2", amount, from_id)
            await conn.execute("UPDATE users SET balance = balance + $1 WHERE id = $2", amount, to_id)
```

---

### GitHub Actions — CI/CD for Agent Projects

**What CI/CD does for your agent**:
- Every push to `main` → automatic test run → catch regressions immediately
- Successful tests → automatic Docker build → catch build failures before deployment
- Successful build → automatic deployment to staging → verify it works
- All automatically, without manual intervention

**Key workflow patterns**:
```yaml
# Caching dependencies — critical for fast CI (30s vs 2min)
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: pip  # cache pip packages between runs

# Testing with secrets — never hardcode API keys
- name: Run tests
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}  # stored in repo Settings > Secrets
  run: pytest tests/ -v

# Docker caching — avoid rebuilding every layer from scratch
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha  # cache from GitHub Actions cache
    cache-to: type=gha,mode=max
```

---

## Common Pitfalls — Week 11

| Mistake | Symptom | Fix |
|---------|---------|-----|
| MCP server dies silently | Client hangs waiting for response | Add `asyncio.wait_for(session.call_tool(...), timeout=30)` |
| asyncpg pool not initialized | `pool is None` AttributeError | Initialize pool in FastAPI `startup` event |
| SQL without parameterization | SQL injection vulnerability | ALWAYS use `$1, $2` placeholders, never f-strings in SQL |
| Not closing pool on shutdown | Resource leak, DB connection exhaustion | Close pool in FastAPI `shutdown` event |
| GitHub Actions secrets in PR from fork | CI fails for external contributors | Use `if: github.event_name != 'pull_request_target'` condition |
- `ex2_mcp_client.py` — agent that connects to an MCP server
- `ex3_postgres_agent.py` — agent that reads/writes PostgreSQL with asyncpg
- `ex4_github_actions.yml` — CI pipeline: test → lint → Docker build → deploy

## Checklist
- [ ] MCP server running, exposing 3+ tools over stdio
- [ ] Agent connects to MCP server and calls tools
- [ ] PostgreSQL schema created: users, sessions, agent_runs, memories
- [ ] Agent run recorded in PostgreSQL on every execution
- [ ] GitHub Actions CI runs tests on every push to main
- [ ] Docker image builds successfully in CI
