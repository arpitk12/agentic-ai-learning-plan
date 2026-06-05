"""
SOLUTION — Exercise 2: Database Schema for Agent Observability
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, calc_cost, MODEL

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "agent_log.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id           TEXT PRIMARY KEY,
            user_id          TEXT,
            session_id       TEXT,
            start_time       TEXT,
            end_time         TEXT,
            status           TEXT DEFAULT 'running',
            input            TEXT,
            output           TEXT,
            model            TEXT,
            total_tokens_in  INTEGER DEFAULT 0,
            total_tokens_out INTEGER DEFAULT 0,
            total_cost_usd   REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            call_id     TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
            tool_name   TEXT,
            arguments   TEXT,
            result      TEXT,
            duration_ms INTEGER,
            called_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_memory (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT,
            updated_at TEXT,
            UNIQUE(user_id, key)
        );
    """)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_run_start(conn: sqlite3.Connection, user_id: str, session_id: str, user_input: str) -> str:
    run_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO agent_runs (run_id, user_id, session_id, start_time, status, input, model)
           VALUES (?, ?, ?, ?, 'running', ?, ?)""",
        (run_id, user_id, session_id, _now(), user_input, MODEL),
    )
    conn.commit()
    return run_id


def log_tool_call(conn: sqlite3.Connection, run_id: str, tool_name: str,
                  arguments: dict, result: str, duration_ms: int) -> None:
    conn.execute(
        """INSERT INTO tool_calls (call_id, run_id, tool_name, arguments, result, duration_ms, called_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), run_id, tool_name, json.dumps(arguments), result, duration_ms, _now()),
    )
    conn.commit()


def log_run_end(conn: sqlite3.Connection, run_id: str, output: str,
                status: str = "completed", tokens_in: int = 0,
                tokens_out: int = 0, cost_usd: float = 0.0) -> None:
    conn.execute(
        """UPDATE agent_runs
           SET end_time=?, status=?, output=?, total_tokens_in=?, total_tokens_out=?, total_cost_usd=?
           WHERE run_id=?""",
        (_now(), status, output, tokens_in, tokens_out, cost_usd, run_id),
    )
    conn.commit()


def store_memory(conn: sqlite3.Connection, user_id: str, key: str, value: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO agent_memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)""",
        (user_id, key, value, _now()),
    )
    conn.commit()


def retrieve_memory(conn: sqlite3.Connection, user_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT key, value FROM agent_memory WHERE user_id=?", (user_id,)
    ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def get_run_summary(conn: sqlite3.Connection, run_id: str) -> dict:
    run = conn.execute("SELECT * FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
    if not run:
        return {}
    calls = conn.execute(
        "SELECT * FROM tool_calls WHERE run_id=? ORDER BY called_at", (run_id,)
    ).fetchall()
    return {
        **dict(run),
        "tool_calls": [dict(c) for c in calls],
    }


def get_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM agent_runs ORDER BY start_time DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


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
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
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
    import time
    conn = get_conn()
    create_schema(conn)

    memories = retrieve_memory(conn, user_id)
    memory_ctx = "\n".join(f"  {k}: {v}" for k, v in memories.items()) if memories else "  (none)"
    system = f"You are a helpful assistant.\n\nUser memories:\n{memory_ctx}"

    run_id = log_run_start(conn, user_id, session_id, question)
    print(f"[Run {run_id[:8]}] Starting: {question}")

    messages = [{"role": "user", "content": question}]
    tokens_in = tokens_out = 0
    final_answer = ""

    for _ in range(5):
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

    summary = get_run_summary(conn, run_id)
    print(f"\n[Summary]\n{json.dumps(summary, indent=2)}")
    conn.close()


if __name__ == "__main__":
    run_logged_agent("Calculate 42 * 73. Also remember my name is Alex.")
    print("\n=== Recent Runs ===")
    conn = get_conn()
    for r in get_recent_runs(conn, limit=5):
        print(f"  {r.get('run_id', '')[:8]}... | {r.get('status')} | {r.get('input', '')[:40]}")
    conn.close()
