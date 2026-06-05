"""
Exercise 2: Database Schema for Agent Observability
Goal: Design and implement a SQLite schema to log all agent activity.

Why this matters:
  Production agents must log every run for debugging, billing, auditing, and
  improvement. A good schema enables queries like:
    - "What tools did the agent use for run XYZ?"
    - "What did the agent remember from user session ABC?"
    - "How much did each run cost this month?"

Tables:
  agent_runs    — one row per agent invocation
  tool_calls    — one row per tool call (foreign key → agent_runs)
  agent_memory  — persistent key-value facts per user/session

Tasks:
  1. Complete create_schema() — CREATE TABLE statements with proper constraints.
  2. Complete log_run_start() — INSERT an agent_run, return its run_id.
  3. Complete log_tool_call() — INSERT a tool_call row.
  4. Complete log_run_end() — UPDATE the run's end_time, status, total_cost, output.
  5. Complete get_run_summary() — SELECT a run + all its tool calls, return dict.
  6. Complete store_memory() / retrieve_memory() — UPSERT / SELECT from agent_memory.
  7. Run a demo agent that logs a full session.

Schema diagram:
  agent_runs ──< tool_calls
  agent_memory (standalone, keyed by user_id + key)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "agent_log.db")

# ── Schema ────────────────────────────────────────────────────────────────────

CREATE_SCHEMA_SQL = """
-- TODO: write the CREATE TABLE IF NOT EXISTS statements

-- agent_runs: one row per agent invocation
-- Columns needed:
--   run_id TEXT PRIMARY KEY
--   user_id TEXT
--   session_id TEXT
--   start_time TEXT (ISO 8601)
--   end_time TEXT
--   status TEXT (running | completed | failed)
--   input TEXT (the user's question)
--   output TEXT (final answer)
--   model TEXT
--   total_tokens_in INTEGER DEFAULT 0
--   total_tokens_out INTEGER DEFAULT 0
--   total_cost_usd REAL DEFAULT 0.0

-- tool_calls: one row per tool call within a run
-- Columns needed:
--   call_id TEXT PRIMARY KEY
--   run_id TEXT (FOREIGN KEY → agent_runs.run_id)
--   tool_name TEXT
--   arguments TEXT (JSON)
--   result TEXT
--   duration_ms INTEGER
--   called_at TEXT (ISO 8601)

-- agent_memory: persistent key-value store per user
-- Columns needed:
--   id INTEGER PRIMARY KEY AUTOINCREMENT
--   user_id TEXT NOT NULL
--   key TEXT NOT NULL
--   value TEXT
--   updated_at TEXT
--   UNIQUE(user_id, key)
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """
    TODO: Execute CREATE_SCHEMA_SQL to set up all three tables.
    Replace the placeholder SQL above with real CREATE TABLE statements first.
    """
    raise NotImplementedError


# ── Run Logging ───────────────────────────────────────────────────────────────

def log_run_start(conn: sqlite3.Connection, user_id: str, session_id: str, user_input: str) -> str:
    """
    INSERT a new row into agent_runs with status='running'.
    TODO:
    1. Generate run_id = str(uuid.uuid4())
    2. INSERT the row (run_id, user_id, session_id, start_time=now, status='running', input, model=MODEL)
    3. conn.commit()
    4. Return run_id
    """
    raise NotImplementedError


def log_tool_call(
    conn: sqlite3.Connection,
    run_id: str,
    tool_name: str,
    arguments: dict,
    result: str,
    duration_ms: int,
) -> None:
    """
    INSERT a row into tool_calls.
    TODO:
    1. call_id = str(uuid.uuid4())
    2. INSERT with called_at=now, arguments=json.dumps(arguments)
    3. conn.commit()
    """
    raise NotImplementedError


def log_run_end(
    conn: sqlite3.Connection,
    run_id: str,
    output: str,
    status: str = "completed",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """
    UPDATE the agent_runs row for run_id.
    TODO: SET end_time=now, status, output, total_tokens_in, total_tokens_out, total_cost_usd
    """
    raise NotImplementedError


# ── Memory ────────────────────────────────────────────────────────────────────

def store_memory(conn: sqlite3.Connection, user_id: str, key: str, value: str) -> None:
    """
    Upsert a memory fact for a user.
    TODO: INSERT OR REPLACE INTO agent_memory (user_id, key, value, updated_at)
    """
    raise NotImplementedError


def retrieve_memory(conn: sqlite3.Connection, user_id: str) -> dict[str, str]:
    """
    Return all memory facts for a user as {key: value}.
    TODO: SELECT key, value FROM agent_memory WHERE user_id=? → dict
    """
    raise NotImplementedError


# ── Queries ───────────────────────────────────────────────────────────────────

def get_run_summary(conn: sqlite3.Connection, run_id: str) -> dict:
    """
    Return a summary dict with:
      - run metadata from agent_runs
      - list of tool_calls for this run
    TODO: Two SELECTs joined by run_id.
    """
    raise NotImplementedError


def get_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """
    Return the most recent `limit` runs (newest first).
    TODO: SELECT * FROM agent_runs ORDER BY start_time DESC LIMIT ?
    """
    raise NotImplementedError


# ── Demo Agent ────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Store a fact in memory for this user",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
]


def execute_tool(name: str, args: dict, conn: sqlite3.Connection, user_id: str) -> str:
    if name == "calculate":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
    elif name == "remember_fact":
        store_memory(conn, user_id, args["key"], args["value"])
        return f"Remembered: {args['key']} = {args['value']}"
    return f"Unknown tool: {name}"


def run_logged_agent(question: str, user_id: str = "demo_user", session_id: str = "demo_session"):
    conn = get_conn()
    create_schema(conn)

    # Load user memory
    memories = retrieve_memory(conn, user_id)
    memory_ctx = "\n".join(f"  {k}: {v}" for k, v in memories.items()) if memories else "  (none)"
    system = f"You are a helpful assistant.\n\nUser memories:\n{memory_ctx}"

    run_id = log_run_start(conn, user_id, session_id, question)
    print(f"[Run {run_id[:8]}] Starting: {question}")

    messages = [{"role": "user", "content": question}]
    tokens_in = tokens_out = 0
    final_answer = ""

    import time
    for _ in range(5):  # max iterations
        response = chat(messages, system=system, tools=TOOLS, max_tokens=512)

        usage = getattr(response, "usage", None)
        if usage:
            tokens_in += getattr(usage, "prompt_tokens", 0)
            tokens_out += getattr(usage, "completion_tokens", 0)

        if stop_reason(response) == "end_turn":
            final_answer = get_text(response)
            break

        messages.append(assistant_message(response))
        for tc in get_tool_calls(response):
            t0 = time.monotonic()
            result = execute_tool(tc["name"], tc["arguments"], conn, user_id)
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_tool_call(conn, run_id, tc["name"], tc["arguments"], result, duration_ms)
            print(f"  🔧 {tc['name']}({tc['arguments']}) → {result}")
            messages.append(tool_result_message(tc["id"], result))

    cost = calc_cost(MODEL, tokens_in, tokens_out)
    log_run_end(conn, run_id, final_answer, "completed", tokens_in, tokens_out, cost)

    print(f"\n[Answer] {final_answer}")
    print(f"[Cost]   ${cost:.6f}  ({tokens_in}in + {tokens_out}out tokens)")

    # Print summary
    summary = get_run_summary(conn, run_id)
    print(f"\n[Summary]\n{json.dumps(summary, indent=2)}")
    conn.close()


if __name__ == "__main__":
    run_logged_agent("Calculate 42 * 73. Also remember my name is Alex.")
    print("\n=== Recent Runs ===")
    conn = get_conn()
    runs = get_recent_runs(conn, limit=5)
    for r in runs:
        print(f"  {r}")
    conn.close()
