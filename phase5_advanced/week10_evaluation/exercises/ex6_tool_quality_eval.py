"""
Exercise 6: Tool Quality Evaluation
Goal: Measure how accurately your agent selects and calls tools.

Theory (from §12.2.3 of the Production Agent Guide):
  - Tool Selection Accuracy — did the agent pick the right tool?    (target > 95%)
  - Argument Validity Rate  — were args syntactically correct?       (target > 92%)
  - Tool Success Rate       — did the tool call return a valid result?(target > 95%)
  - Unnecessary Call Rate   — fraction of redundant/wrong calls      (target < 8%)

Architecture:
  Instead of calling a real LLM agent we use a *mock agent*:
  - You give the agent a list of available tools (with descriptions).
  - The agent (via LLM) chooses which tool to call and with what arguments.
  - A MockDispatcher records every call and validates against the expected schema.
  - After all test cases you compute the four metrics.

Tasks:
  1. Complete validate_args()     — check if args match the tool's Pydantic schema.
  2. Complete MockDispatcher.call() — record call, validate args, return mock result.
  3. Complete run_tool_eval()     — run all cases, collect CallRecord per case.
  4. Complete compute_metrics()   — calculate the four metrics from CallRecord list.
  5. Add 3 more test cases to TOOL_EVAL_CASES covering tool selection errors.

Run:
  python ex6_tool_quality_eval.py

Expected output (rough):
  Tool Selection Accuracy : 93.3%  ✅ (target ≥ 95%)
  Argument Validity Rate  : 96.7%  ✅ (target ≥ 92%)
  Tool Success Rate       : 100.0% ✅ (target ≥ 95%)
  Unnecessary Call Rate   :  6.7%  ✅ (target ≤ 8%)
"""

import os, sys, re, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

# ── Tool schemas (Pydantic models = ground truth for arg validation) ────────────

class WebSearchArgs(BaseModel):
    query: str
    max_results: int = 5

class ReadFileArgs(BaseModel):
    path: str
    encoding: str = "utf-8"

class WriteFileArgs(BaseModel):
    path: str
    content: str

class RunPythonArgs(BaseModel):
    code: str
    timeout: int = 10

class SendEmailArgs(BaseModel):
    to: str
    subject: str
    body: str


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "web_search":  WebSearchArgs,
    "read_file":   ReadFileArgs,
    "write_file":  WriteFileArgs,
    "run_python":  RunPythonArgs,
    "send_email":  SendEmailArgs,
}

TOOL_DESCRIPTIONS = {
    "web_search":  "Search the web for current information. Args: query (str), max_results (int, optional).",
    "read_file":   "Read the contents of a local file. Args: path (str), encoding (str, optional).",
    "write_file":  "Write text content to a local file. Args: path (str), content (str).",
    "run_python":  "Execute a Python code snippet and return stdout. Args: code (str), timeout (int, optional).",
    "send_email":  "Send an email. Args: to (str), subject (str), body (str).",
}


# ── Test case schema ───────────────────────────────────────────────────────────

@dataclass
class ToolEvalCase:
    id:            str
    user_request:  str
    expected_tool: str              # which tool SHOULD be called
    expected_args_keys: list[str]   # required arg keys that must be present
    should_not_call: list[str] = field(default_factory=list)  # tools that should NOT be called


@dataclass
class CallRecord:
    case_id:        str
    selected_tool:  str
    args:           dict
    args_valid:     bool
    correct_tool:   bool
    success:        bool
    unnecessary:    bool   # called a wrong / redundant tool


# ── Test corpus ────────────────────────────────────────────────────────────────

TOOL_EVAL_CASES: list[ToolEvalCase] = [
    ToolEvalCase("t001", "Find the latest news about LLMs.",
                 expected_tool="web_search", expected_args_keys=["query"],
                 should_not_call=["read_file", "send_email"]),

    ToolEvalCase("t002", "Read the file at /tmp/report.txt",
                 expected_tool="read_file", expected_args_keys=["path"],
                 should_not_call=["web_search", "run_python"]),

    ToolEvalCase("t003", "Save 'Hello World' to /tmp/hello.txt",
                 expected_tool="write_file", expected_args_keys=["path", "content"],
                 should_not_call=["read_file", "web_search"]),

    ToolEvalCase("t004", "Calculate the factorial of 12 using Python.",
                 expected_tool="run_python", expected_args_keys=["code"],
                 should_not_call=["web_search", "send_email"]),

    ToolEvalCase("t005", "Send an email to alice@example.com about the deployment.",
                 expected_tool="send_email", expected_args_keys=["to", "subject", "body"],
                 should_not_call=["web_search", "run_python"]),

    ToolEvalCase("t006", "What is the capital of Australia?",
                 expected_tool="web_search", expected_args_keys=["query"],
                 should_not_call=["run_python", "send_email"]),

    ToolEvalCase("t007", "Write a Python script that prints the first 10 Fibonacci numbers and run it.",
                 expected_tool="run_python", expected_args_keys=["code"],
                 should_not_call=["send_email", "read_file"]),

    ToolEvalCase("t008", "Load the configuration from /etc/myapp/config.yaml",
                 expected_tool="read_file", expected_args_keys=["path"],
                 should_not_call=["write_file", "send_email"]),

    ToolEvalCase("t009", "Search for 'asyncio tutorial' and save the results to /tmp/results.txt",
                 expected_tool="web_search", expected_args_keys=["query"],
                 should_not_call=["send_email", "run_python"]),

    ToolEvalCase("t010", "Email bob@example.com the summary: 'Deployment complete.'",
                 expected_tool="send_email", expected_args_keys=["to", "subject", "body"],
                 should_not_call=["run_python", "read_file"]),

    # TODO: Add 3 more cases that test edge cases or tool selection errors.
    # Example ideas:
    # - "Compute 2+2" — should use run_python, not web_search
    # - "Look up today's weather" — should use web_search, not run_python
    # - "Create a backup of /data/db.sql and email it to ops@company.com" — multi-step
]


# ── Agent that selects and calls a tool ────────────────────────────────────────

AGENT_SYSTEM = """\
You are a tool-using AI assistant. Given a user request and a list of available tools,
respond with a JSON object describing the ONE best tool call to make:
{"tool": "<tool_name>", "args": {<key>: <value>, ...}}

Available tools:
""" + "\n".join(f"  - {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) + """

Respond ONLY with valid JSON. No markdown, no explanation."""


async def agent_select_tool(request: str) -> tuple[str, dict]:
    """Ask the LLM to select a tool and provide args. Returns (tool_name, args_dict)."""
    messages = [{"role": "user", "content": request}]
    response = await achat(messages, system=AGENT_SYSTEM, max_tokens=300)
    text = get_text(response).strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(text)
    return parsed.get("tool", ""), parsed.get("args", {})


# ── Mock dispatcher ────────────────────────────────────────────────────────────

MOCK_RESULTS = {
    "web_search":  lambda args: {"results": [f"Result for: {args.get('query', '')}"]},
    "read_file":   lambda args: {"content": f"<contents of {args.get('path', '')}>"},
    "write_file":  lambda args: {"written": args.get("path", "")},
    "run_python":  lambda args: {"stdout": "42", "stderr": ""},
    "send_email":  lambda args: {"sent": True, "to": args.get("to", "")},
}


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Complete validate_args()
# ─────────────────────────────────────────────────────────────────────────────

def validate_args(tool_name: str, args: dict) -> bool:
    """
    Return True if args satisfy the Pydantic schema for tool_name.

    TODO:
    1. Look up TOOL_SCHEMAS[tool_name]. If not found, return False.
    2. Try to instantiate the schema with **args.
    3. Return True if it succeeds, False if ValidationError is raised.

    Hint: use try/except pydantic.ValidationError.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete run_tool_eval()
# ─────────────────────────────────────────────────────────────────────────────

async def run_tool_eval(cases: list[ToolEvalCase]) -> list[CallRecord]:
    """
    Run every test case through the agent, record results.

    For each case:
      - Call agent_select_tool(case.user_request) → (selected_tool, args)
      - Wrap in try/except json.JSONDecodeError → treat as ("", {}) on parse failure
      - correct_tool   = (selected_tool == case.expected_tool)
      - args_valid     = validate_args(selected_tool, args)
      - success        = correct_tool and args_valid
      - unnecessary    = selected_tool in case.should_not_call
      - Append a CallRecord to results.

    Run all cases concurrently with asyncio.gather().

    TODO: implement using asyncio.gather + list comprehension.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete compute_metrics()
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolMetrics:
    tool_selection_accuracy: float
    arg_validity_rate:       float
    tool_success_rate:       float
    unnecessary_call_rate:   float
    n:                       int


def compute_metrics(records: list[CallRecord]) -> ToolMetrics:
    """
    Compute the four quality metrics from a list of CallRecords.

    Definitions:
      tool_selection_accuracy = # correct_tool / total
      arg_validity_rate       = # args_valid   / total
      tool_success_rate       = # success      / total
      unnecessary_call_rate   = # unnecessary  / total

    TODO: implement using sum() + len().
    """
    raise NotImplementedError


# ── Report ─────────────────────────────────────────────────────────────────────

METRIC_TARGETS = {
    "tool_selection_accuracy": (0.95, "≥"),
    "arg_validity_rate":       (0.92, "≥"),
    "tool_success_rate":       (0.95, "≥"),
    "unnecessary_call_rate":   (0.08, "≤"),
}

def print_report(metrics: ToolMetrics, records: list[CallRecord]):
    print(f"\nTool Quality Evaluation — {metrics.n} test cases\n")
    print(f"  {'Tool Selection Accuracy':<28}: {metrics.tool_selection_accuracy:6.1%}  "
          f"{'✅' if metrics.tool_selection_accuracy >= 0.95 else '❌'}  (target ≥ 95%)")
    print(f"  {'Argument Validity Rate':<28}: {metrics.arg_validity_rate:6.1%}  "
          f"{'✅' if metrics.arg_validity_rate >= 0.92 else '❌'}  (target ≥ 92%)")
    print(f"  {'Tool Success Rate':<28}: {metrics.tool_success_rate:6.1%}  "
          f"{'✅' if metrics.tool_success_rate >= 0.95 else '❌'}  (target ≥ 95%)")
    print(f"  {'Unnecessary Call Rate':<28}: {metrics.unnecessary_call_rate:6.1%}  "
          f"{'✅' if metrics.unnecessary_call_rate <= 0.08 else '❌'}  (target ≤ 8%)")

    failures = [r for r in records if not r.correct_tool or not r.args_valid]
    if failures:
        print("\n  Failures:")
        for r in failures[:5]:
            print(f"    [{r.case_id}] selected={r.selected_tool!r}  args_valid={r.args_valid}  "
                  f"args={json.dumps(r.args)[:60]}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("TOOL QUALITY EVALUATION")
    print("=" * 60)
    records = await run_tool_eval(TOOL_EVAL_CASES)
    metrics = compute_metrics(records)
    print_report(metrics, records)


if __name__ == "__main__":
    asyncio.run(main())
