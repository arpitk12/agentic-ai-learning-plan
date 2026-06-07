"""
Solution 6: Tool Quality Evaluation
"""

import os, sys, re, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()


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
    "web_search": WebSearchArgs,
    "read_file":  ReadFileArgs,
    "write_file": WriteFileArgs,
    "run_python": RunPythonArgs,
    "send_email": SendEmailArgs,
}

TOOL_DESCRIPTIONS = {
    "web_search":  "Search the web for current information. Args: query (str), max_results (int, optional).",
    "read_file":   "Read the contents of a local file. Args: path (str), encoding (str, optional).",
    "write_file":  "Write text content to a local file. Args: path (str), content (str).",
    "run_python":  "Execute a Python code snippet and return stdout. Args: code (str), timeout (int, optional).",
    "send_email":  "Send an email. Args: to (str), subject (str), body (str).",
}


@dataclass
class ToolEvalCase:
    id:            str
    user_request:  str
    expected_tool: str
    expected_args_keys: list[str]
    should_not_call: list[str] = field(default_factory=list)


@dataclass
class CallRecord:
    case_id:       str
    selected_tool: str
    args:          dict
    args_valid:    bool
    correct_tool:  bool
    success:       bool
    unnecessary:   bool


@dataclass
class ToolMetrics:
    tool_selection_accuracy: float
    arg_validity_rate:       float
    tool_success_rate:       float
    unnecessary_call_rate:   float
    n:                       int


TOOL_EVAL_CASES: list[ToolEvalCase] = [
    ToolEvalCase("t001", "Find the latest news about LLMs.",
                 "web_search", ["query"], ["read_file", "send_email"]),
    ToolEvalCase("t002", "Read the file at /tmp/report.txt",
                 "read_file", ["path"], ["web_search", "run_python"]),
    ToolEvalCase("t003", "Save 'Hello World' to /tmp/hello.txt",
                 "write_file", ["path", "content"], ["read_file", "web_search"]),
    ToolEvalCase("t004", "Calculate the factorial of 12 using Python.",
                 "run_python", ["code"], ["web_search", "send_email"]),
    ToolEvalCase("t005", "Send an email to alice@example.com about the deployment.",
                 "send_email", ["to", "subject", "body"], ["web_search", "run_python"]),
    ToolEvalCase("t006", "What is the capital of Australia?",
                 "web_search", ["query"], ["run_python", "send_email"]),
    ToolEvalCase("t007", "Write a Python script that prints the first 10 Fibonacci numbers and run it.",
                 "run_python", ["code"], ["send_email", "read_file"]),
    ToolEvalCase("t008", "Load the configuration from /etc/myapp/config.yaml",
                 "read_file", ["path"], ["write_file", "send_email"]),
    ToolEvalCase("t009", "Search for 'asyncio tutorial' and save the results to /tmp/results.txt",
                 "web_search", ["query"], ["send_email", "run_python"]),
    ToolEvalCase("t010", "Email bob@example.com the summary: 'Deployment complete.'",
                 "send_email", ["to", "subject", "body"], ["run_python", "read_file"]),
    ToolEvalCase("t011", "Compute 2 + 2",
                 "run_python", ["code"], ["web_search", "send_email"]),
    ToolEvalCase("t012", "What is today's weather in London?",
                 "web_search", ["query"], ["run_python", "read_file"]),
    ToolEvalCase("t013", "Append 'done' to /tmp/log.txt",
                 "write_file", ["path", "content"], ["web_search", "send_email"]),
]


AGENT_SYSTEM = """\
You are a tool-using AI assistant. Given a user request and a list of available tools,
respond with a JSON object describing the ONE best tool call to make:
{"tool": "<tool_name>", "args": {<key>: <value>, ...}}

Available tools:
""" + "\n".join(f"  - {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) + """

Respond ONLY with valid JSON. No markdown, no explanation."""


async def agent_select_tool(request: str) -> tuple[str, dict]:
    messages = [{"role": "user", "content": request}]
    response = await achat(messages, system=AGENT_SYSTEM, max_tokens=300)
    text = get_text(response).strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(text)
    return parsed.get("tool", ""), parsed.get("args", {})


# ── Solution implementations ───────────────────────────────────────────────────

def validate_args(tool_name: str, args: dict) -> bool:
    """Return True if args satisfy the Pydantic schema for tool_name."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return False
    try:
        schema(**args)
        return True
    except ValidationError:
        return False


async def _eval_one(case: ToolEvalCase) -> CallRecord:
    try:
        selected_tool, args = await agent_select_tool(case.user_request)
    except (json.JSONDecodeError, KeyError):
        selected_tool, args = "", {}

    correct_tool = (selected_tool == case.expected_tool)
    args_valid   = validate_args(selected_tool, args)
    success      = correct_tool and args_valid
    unnecessary  = selected_tool in case.should_not_call

    return CallRecord(
        case_id=case.id,
        selected_tool=selected_tool,
        args=args,
        args_valid=args_valid,
        correct_tool=correct_tool,
        success=success,
        unnecessary=unnecessary,
    )


async def run_tool_eval(cases: list[ToolEvalCase]) -> list[CallRecord]:
    """Run all cases concurrently and collect CallRecords."""
    return list(await asyncio.gather(*[_eval_one(c) for c in cases]))


def compute_metrics(records: list[CallRecord]) -> ToolMetrics:
    """Compute the four quality metrics from CallRecords."""
    n = len(records)
    if n == 0:
        return ToolMetrics(0, 0, 0, 0, 0)
    return ToolMetrics(
        tool_selection_accuracy=sum(r.correct_tool  for r in records) / n,
        arg_validity_rate=      sum(r.args_valid    for r in records) / n,
        tool_success_rate=      sum(r.success       for r in records) / n,
        unnecessary_call_rate=  sum(r.unnecessary   for r in records) / n,
        n=n,
    )


def print_report(metrics: ToolMetrics, records: list[CallRecord]):
    print(f"\nTool Quality Evaluation — {metrics.n} test cases\n")
    rows = [
        ("Tool Selection Accuracy", metrics.tool_selection_accuracy, 0.95, "≥"),
        ("Argument Validity Rate",  metrics.arg_validity_rate,       0.92, "≥"),
        ("Tool Success Rate",       metrics.tool_success_rate,       0.95, "≥"),
        ("Unnecessary Call Rate",   metrics.unnecessary_call_rate,   0.08, "≤"),
    ]
    for label, value, target, direction in rows:
        ok   = value >= target if direction == "≥" else value <= target
        icon = "✅" if ok else "❌"
        print(f"  {label:<28}: {value:6.1%}  {icon}  (target {direction} {target:.0%})")

    failures = [r for r in records if not r.correct_tool or not r.args_valid]
    if failures:
        print("\n  Failures:")
        for r in failures[:5]:
            print(f"    [{r.case_id}] selected={r.selected_tool!r}  "
                  f"args_valid={r.args_valid}  args={json.dumps(r.args)[:60]}")


async def main():
    print("=" * 60)
    print("TOOL QUALITY EVALUATION")
    print("=" * 60)
    records = await run_tool_eval(TOOL_EVAL_CASES)
    metrics = compute_metrics(records)
    print_report(metrics, records)


if __name__ == "__main__":
    asyncio.run(main())
