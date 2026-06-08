"""
solution/src/observability/cost_tracker.py — Full implementation.
"""
from __future__ import annotations
import sqlite3
import time
import uuid

COST_PER_1K: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":        (0.005,   0.015),
    "openai/gpt-4o-mini":   (0.00015, 0.0006),
    "openai/gpt-3.5-turbo": (0.0005,  0.0015),
    "openai/gpt-4-turbo":   (0.01,    0.03),
}
_DEFAULT_RATES = (0.00015, 0.0006)


class CostTracker:
    def __init__(self, db_path: str = "/tmp/agent_costs.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                ts REAL DEFAULT (unixepoch())
            )
        """)
        self._conn.commit()

    def record(self, user_id: str, model: str,
               input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = COST_PER_1K.get(model, _DEFAULT_RATES)
        cost = (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)
        self._conn.execute(
            "INSERT INTO calls (id, user_id, model, input_tokens, output_tokens, cost_usd) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, model, input_tokens, output_tokens, cost),
        )
        self._conn.commit()
        return cost

    def get_summary(self, user_id: str | None = None, since_hours: float = 24.0) -> dict:
        cutoff = time.time() - since_hours * 3600
        params: list = [cutoff]
        where = "WHERE ts > ?"
        if user_id:
            where += " AND user_id = ?"
            params.append(user_id)
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(cost_usd),0), COALESCE(SUM(input_tokens),0), "
            f"COALESCE(SUM(output_tokens),0), COUNT(*) FROM calls {where}",
            params,
        ).fetchone()
        return {
            "total_cost_usd": row[0],
            "total_input_tokens": row[1],
            "total_output_tokens": row[2],
            "calls": row[3],
            "window_hours": since_hours,
        }

    def get_total(self, user_id: str | None = None) -> float:
        if user_id:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM calls WHERE user_id=?", (user_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM calls").fetchone()
        return row[0] if row else 0.0
