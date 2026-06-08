"""SqliteSaver checkpointer factory — enables interrupt/resume and state persistence."""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "checkpoints.db"


def get_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    """Return a SqliteSaver checkpointer.

    Args:
        db_path: Path to the SQLite database file. Defaults to data/checkpoints.db.

    Returns:
        SqliteSaver instance (use as a context manager for cleanup).

    Usage::

        with get_checkpointer() as cp:
            graph = build_graph(checkpointer=cp)
            result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})
    """
    path = Path(db_path) if db_path else _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(path))
