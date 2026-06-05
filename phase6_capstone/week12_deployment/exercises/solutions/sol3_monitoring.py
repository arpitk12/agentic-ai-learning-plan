"""
SOLUTION — Exercise 3: Prometheus Metrics + Grafana Setup
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import time
import json
import asyncio
from dotenv import load_dotenv
from llm import achat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("prometheus_client not installed. Run: pip install prometheus-client")


def setup_metrics():
    if not PROMETHEUS_AVAILABLE:
        return None
    return {
        "requests_total": Counter(
            "agent_requests_total", "Total agent API requests", ["status", "model"]
        ),
        "latency_seconds": Histogram(
            "agent_latency_seconds", "Agent response latency", ["model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        ),
        "active_runs": Gauge(
            "active_agent_runs", "Number of agent runs currently in progress"
        ),
        "tool_calls_total": Counter(
            "tool_calls_total", "Total tool calls by tool name", ["tool_name"]
        ),
        "llm_tokens_total": Counter(
            "llm_tokens_total", "Total LLM tokens consumed", ["direction", "model"]
        ),
        "llm_cost_total": Counter(
            "llm_cost_usd_total", "Total LLM cost in USD", ["model"]
        ),
    }


METRICS = setup_metrics()


def increment_request(status: str, model: str = MODEL) -> None:
    if not METRICS:
        return
    METRICS["requests_total"].labels(status=status, model=model).inc()


def record_latency(duration: float, model: str = MODEL) -> None:
    if not METRICS:
        return
    METRICS["latency_seconds"].labels(model=model).observe(duration)


def track_tool_call(tool_name: str) -> None:
    if not METRICS:
        return
    METRICS["tool_calls_total"].labels(tool_name=tool_name).inc()


def track_tokens(input_tokens: int, output_tokens: int, model: str = MODEL) -> None:
    if not METRICS:
        return
    METRICS["llm_tokens_total"].labels(direction="input", model=model).inc(input_tokens)
    METRICS["llm_tokens_total"].labels(direction="output", model=model).inc(output_tokens)
    cost = calc_cost(model, input_tokens, output_tokens)
    METRICS["llm_cost_total"].labels(model=model).inc(cost)


try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Monitored Agent API", version="1.0.0")

    class AgentRequest(BaseModel):
        message: str
        session_id: str = "default"
        max_tokens: int = 512

    class AgentResponse(BaseModel):
        answer: str
        tokens_in: int
        tokens_out: int
        cost_usd: float
        latency_seconds: float

    TOOLS = [
        {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    async def run_tool(name: str, args: dict) -> str:
        track_tool_call(name)
        if name == "calculator":
            try:
                return str(eval(args["expression"], {"__builtins__": {}}))
            except Exception as e:
                return f"Error: {e}"
        return f"Unknown tool: {name}"

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": MODEL}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        if not PROMETHEUS_AVAILABLE:
            return PlainTextResponse("prometheus_client not installed", status_code=503)
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.post("/agent", response_model=AgentResponse)
    async def agent_endpoint(req: AgentRequest):
        if METRICS:
            METRICS["active_runs"].inc()
        start = time.monotonic()
        status = "error"
        tokens_in = tokens_out = 0
        answer = ""
        try:
            messages = [{"role": "user", "content": req.message}]
            for _ in range(8):
                response = await achat(messages, tools=TOOLS, max_tokens=req.max_tokens)
                usage = getattr(response, "usage", None)
                if usage:
                    tokens_in += getattr(usage, "prompt_tokens", 0)
                    tokens_out += getattr(usage, "completion_tokens", 0)

                if stop_reason(response) == "end_turn":
                    answer = get_text(response)
                    break

                messages.append(assistant_message(response))
                for tc in get_tool_calls(response):
                    result = await run_tool(tc["name"], tc["arguments"])
                    messages.append(tool_result_message(tc["id"], result))

            status = "success"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            duration = time.monotonic() - start
            if METRICS:
                METRICS["active_runs"].dec()
            increment_request(status)
            record_latency(duration)
            track_tokens(tokens_in, tokens_out)

        cost = calc_cost(MODEL, tokens_in, tokens_out)
        return AgentResponse(
            answer=answer, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost, latency_seconds=round(duration, 3),
        )

    @app.get("/admin/metrics-summary")
    async def metrics_summary():
        return {
            "message": "Check /metrics for Prometheus format",
            "grafana_setup": "docker run -p 3000:3000 grafana/grafana",
            "prometheus_scrape_config": {
                "scrape_configs": [{"job_name": "agent-api", "static_configs": [{"targets": ["localhost:8000"]}]}]
            },
        }


def generate_grafana_dashboard() -> dict:
    return {
        "title": "Agent API Dashboard",
        "uid": "agent-api-001",
        "panels": [
            {
                "title": "Request Rate (req/s)",
                "type": "graph",
                "targets": [{"expr": "rate(agent_requests_total[5m])", "legendFormat": "{{status}}"}],
            },
            {
                "title": "P95 Latency (s)",
                "type": "graph",
                "targets": [{"expr": "histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m]))"}],
            },
            {
                "title": "P50 Latency (s)",
                "type": "graph",
                "targets": [{"expr": "histogram_quantile(0.50, rate(agent_latency_seconds_bucket[5m]))"}],
            },
            {
                "title": "Active Runs",
                "type": "stat",
                "targets": [{"expr": "active_agent_runs"}],
            },
            {
                "title": "Token Consumption Rate",
                "type": "graph",
                "targets": [
                    {"expr": "rate(llm_tokens_total{direction='input'}[5m])", "legendFormat": "input"},
                    {"expr": "rate(llm_tokens_total{direction='output'}[5m])", "legendFormat": "output"},
                ],
            },
            {
                "title": "Total Cost (USD)",
                "type": "stat",
                "targets": [{"expr": "llm_cost_usd_total"}],
            },
            {
                "title": "Tool Calls by Type",
                "type": "graph",
                "targets": [{"expr": "rate(tool_calls_total[5m])", "legendFormat": "{{tool_name}}"}],
            },
        ],
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
    }


if __name__ == "__main__":
    dashboard = generate_grafana_dashboard()
    dashboard_path = os.path.join(os.path.dirname(__file__), "grafana_dashboard.json")
    with open(dashboard_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"✓ Grafana dashboard saved to {dashboard_path}")

    if FASTAPI_AVAILABLE:
        import uvicorn
        print("\nStarting monitored agent API on http://localhost:8000")
        print("  Metrics:  http://localhost:8000/metrics")
        print("  Docs:     http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Install FastAPI: pip install fastapi uvicorn prometheus-client")
