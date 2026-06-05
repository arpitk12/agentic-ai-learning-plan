"""
Exercise 3: Prometheus Metrics + Grafana Setup
Goal: Instrument your agent API with production metrics.

Install: pip install prometheus-client fastapi uvicorn

Key metrics to track:
  - agent_requests_total      (Counter)   — total requests by status
  - agent_latency_seconds     (Histogram)  — response time distribution
  - active_agent_runs         (Gauge)      — concurrent requests right now
  - tool_calls_total          (Counter)    — tool usage by tool name
  - llm_tokens_total          (Counter)    — token consumption by direction
  - llm_cost_total            (Counter)    — total $ spent

Run:
  python ex3_monitoring.py
  # Open http://localhost:8000/metrics — Prometheus scrapes this
  # Open http://localhost:8000/docs    — test the API
  # Open http://localhost:3000         — Grafana dashboard (if docker-compose up)

Tasks:
  1. Complete increment_request() — labels: status ("success"|"error"), model.
  2. Complete record_latency() — use the histogram context manager or observe().
  3. Complete track_tool_call() — increment tool_calls_total with tool_name label.
  4. Complete track_tokens() — increment llm_tokens_total for "input" and "output".
  5. Complete the /agent endpoint — use the metrics inside the request handler.
  6. Run the server and verify metrics at /metrics.
  7. (Bonus) Write the Grafana dashboard JSON and save to grafana_dashboard.json.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import time
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from llm import achat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

# ── Prometheus Metrics Setup ──────────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("prometheus_client not installed. Run: pip install prometheus-client")


def setup_metrics():
    """Define all Prometheus metrics. Called once at startup."""
    if not PROMETHEUS_AVAILABLE:
        return None

    metrics = {
        "requests_total": Counter(
            "agent_requests_total",
            "Total agent API requests",
            ["status", "model"],  # labels
        ),
        "latency_seconds": Histogram(
            "agent_latency_seconds",
            "Agent response latency",
            ["model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        ),
        "active_runs": Gauge(
            "active_agent_runs",
            "Number of agent runs currently in progress",
        ),
        "tool_calls_total": Counter(
            "tool_calls_total",
            "Total tool calls by tool name",
            ["tool_name"],
        ),
        "llm_tokens_total": Counter(
            "llm_tokens_total",
            "Total LLM tokens consumed",
            ["direction", "model"],  # direction: "input" or "output"
        ),
        "llm_cost_total": Counter(
            "llm_cost_usd_total",
            "Total LLM cost in USD",
            ["model"],
        ),
    }
    return metrics


METRICS = setup_metrics()


# ── Metric Helpers ────────────────────────────────────────────────────────────

def increment_request(status: str, model: str = MODEL) -> None:
    """
    Increment agent_requests_total with labels status and model.
    TODO: METRICS["requests_total"].labels(status=status, model=model).inc()
    """
    if not METRICS:
        return
    raise NotImplementedError


def record_latency(duration: float, model: str = MODEL) -> None:
    """
    Record a latency sample (seconds) to the histogram.
    TODO: METRICS["latency_seconds"].labels(model=model).observe(duration)
    """
    if not METRICS:
        return
    raise NotImplementedError


def track_tool_call(tool_name: str) -> None:
    """
    Increment tool_calls_total for the given tool_name.
    TODO: one line
    """
    if not METRICS:
        return
    raise NotImplementedError


def track_tokens(input_tokens: int, output_tokens: int, model: str = MODEL) -> None:
    """
    Increment llm_tokens_total for both "input" and "output" directions.
    Also increment llm_cost_total with calc_cost(model, input_tokens, output_tokens).
    TODO: 3 lines
    """
    if not METRICS:
        return
    raise NotImplementedError


# ── FastAPI App ───────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

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
        """Prometheus metrics endpoint — configure Prometheus to scrape this."""
        if not PROMETHEUS_AVAILABLE:
            return PlainTextResponse("prometheus_client not installed", status_code=503)
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/agent", response_model=AgentResponse)
    async def agent_endpoint(req: AgentRequest):
        """
        TODO:
        1. Increment active_runs gauge (+1 at start, -1 at end via try/finally).
        2. Record start time.
        3. Run the agent loop (use achat + tool calls).
        4. Record tokens and latency.
        5. Increment request counter with status "success" or "error".
        6. Return AgentResponse.
        """
        if METRICS:
            METRICS["active_runs"].inc()
        start = time.monotonic()
        status = "error"
        tokens_in = tokens_out = 0
        answer = ""
        try:
            # TODO: implement the agent loop here (reuse pattern from other exercises)
            raise NotImplementedError

            status = "success"
        except NotImplementedError:
            raise HTTPException(status_code=501, detail="Agent loop not implemented yet")
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
            answer=answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_seconds=round(duration, 3),
        )

    @app.get("/admin/metrics-summary")
    async def metrics_summary():
        """Human-readable metrics overview."""
        if not PROMETHEUS_AVAILABLE:
            return {"error": "prometheus_client not installed"}
        # Collect current metric values
        return {
            "message": "Check /metrics for Prometheus format",
            "grafana_setup": "docker run -p 3000:3000 grafana/grafana",
            "prometheus_scrape_config": {
                "scrape_configs": [{
                    "job_name": "agent-api",
                    "static_configs": [{"targets": ["localhost:8000"]}],
                }]
            },
        }


# ── Grafana Dashboard Generator ───────────────────────────────────────────────

def generate_grafana_dashboard() -> dict:
    """
    Generate a basic Grafana dashboard JSON for the agent metrics.
    TODO (Bonus): Add panels for:
      - Request rate over time (requests_total rate)
      - P50/P95/P99 latency (histogram_quantile)
      - Active runs (gauge)
      - Token consumption rate
      - Cost per hour
    """
    return {
        "title": "Agent API Dashboard",
        "panels": [
            {
                "title": "Request Rate",
                "type": "graph",
                "targets": [{"expr": "rate(agent_requests_total[5m])"}],
            },
            {
                "title": "P95 Latency",
                "type": "graph",
                "targets": [{"expr": "histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m]))"}],
            },
            {
                "title": "Active Runs",
                "type": "stat",
                "targets": [{"expr": "active_agent_runs"}],
            },
            # TODO: Add token consumption and cost panels
        ],
        "refresh": "30s",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Save Grafana dashboard
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
        print("  Health:   http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Install FastAPI: pip install fastapi uvicorn prometheus-client")
