# Week 12 Resources — Deploy, Monitor & Optimize

## Container & Orchestration
- Docker docs: https://docs.docker.com/
- Kubernetes for ML: https://kubernetes.io/docs/concepts/workloads/pods/
- Railway (easy deploy): https://railway.app/
- Modal (serverless): https://modal.com/

## Monitoring & Alerting
- Prometheus + Grafana: https://grafana.com/docs/grafana/latest/
- Datadog LLM Monitoring: https://www.datadoghq.com/product/llm-observability/
- PagerDuty for on-call: https://www.pagerduty.com/

## Cost Optimization
- LiteLLM (model routing + cost tracking): https://github.com/BerriAI/litellm
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Batching API (Anthropic): https://docs.anthropic.com/en/docs/build-with-claude/message-batches

## Production Checklist
- [ ] Dockerfile with non-root user
- [ ] Health check endpoint (`/health`)
- [ ] Structured logs to stdout (parsed by log aggregator)
- [ ] Cost per request tracked and alerting at 2x baseline
- [ ] Rollback plan if model degrades
- [ ] Prompt versioning (tag which prompt version each run used)
- [ ] Model router tested to confirm cost savings

## Install
```
pip install anthropic fastapi uvicorn structlog prometheus-client litellm
docker build -t my-agent . && docker run -p 8000:8000 my-agent
```
