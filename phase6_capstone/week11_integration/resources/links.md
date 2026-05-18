# Week 11 Resources — System Integration

## Model Context Protocol (MCP)
- MCP Spec: https://modelcontextprotocol.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Servers (community): https://github.com/modelcontextprotocol/servers

## Database & Persistence
- SQLAlchemy (async): https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Alembic (migrations): https://alembic.sqlalchemy.org/
- Redis for agent state: https://redis.io/docs/latest/

## CI/CD for Agents
- GitHub Actions: https://docs.github.com/en/actions
- Running evals in CI: https://docs.smith.langchain.com/evaluation/tutorials/ci-cd
- pytest for agents: https://docs.pytest.org/

## Install
```
pip install mcp sqlalchemy alembic asyncpg redis pytest
```

## Integration Checklist
- [ ] Agent runs logged to database (not just stdout)
- [ ] MCP tool server with at least 3 tools
- [ ] CI pipeline runs eval suite on every PR
- [ ] Database migrations versioned with Alembic
- [ ] All secrets in environment variables (never hardcoded)
