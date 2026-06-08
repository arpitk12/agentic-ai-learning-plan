"""
src/observability/logger.py — Structured logging with structlog.

TODO:
  1. implement configure_logging() — set up structlog with JSON renderer
  2. implement get_logger() — return a bound structlog logger
"""
from __future__ import annotations


# ── TODO 1: Configure structlog ───────────────────────────────────────────────
# def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
#     """
#     Configure structlog with appropriate processors.
#
#     Processors chain (in order):
#       - structlog.contextvars.merge_contextvars  ← injects request_id etc.
#       - structlog.processors.add_log_level
#       - structlog.processors.TimeStamper(fmt="iso")
#       - structlog.processors.StackInfoRenderer()
#       - structlog.processors.format_exc_info
#       - structlog.processors.JSONRenderer() if json_output
#         else structlog.dev.ConsoleRenderer()
#
#     Steps:
#       1a. import structlog
#       1b. structlog.configure(
#               processors=[...],
#               wrapper_class=structlog.make_filtering_bound_logger(log_level),
#               context_class=dict,
#               logger_factory=structlog.PrintLoggerFactory(),
#           )
#     """
#     raise NotImplementedError


# ── TODO 2: Get logger ────────────────────────────────────────────────────────
# def get_logger(name: str = "agent") -> structlog.BoundLogger:
#     """Return a structlog logger bound with the given name."""
#     import structlog
#     return structlog.get_logger(name)

raise NotImplementedError("Implement configure_logging() and get_logger() in src/observability/logger.py")
