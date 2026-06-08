"""Custom LangChain callback handler for structured logging."""
from __future__ import annotations

import logging
import structlog
from langchain_core.callbacks import BaseCallbackHandler

log = structlog.get_logger()


class LoggingCallbackHandler(BaseCallbackHandler):
    """Logs agent events with structlog for production observability."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        log.debug("llm_start", model=serialized.get("kwargs", {}).get("model_name", "?"))

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output or {}
        log.info("llm_end",
                 tokens=usage.get("token_usage", {}).get("total_tokens"),
                 model=usage.get("model_name"))

    def on_tool_start(self, serialized, input_str, **kwargs):
        log.info("tool_start", tool=serialized.get("name"), input=input_str[:100])

    def on_tool_end(self, output, **kwargs):
        log.info("tool_end", output=str(output)[:100])

    def on_tool_error(self, error, **kwargs):
        log.error("tool_error", error=str(error))

    def on_agent_action(self, action, **kwargs):
        log.info("agent_action", tool=action.tool, input=str(action.tool_input)[:80])

    def on_agent_finish(self, finish, **kwargs):
        log.info("agent_finish", output=str(finish.return_values.get("output", ""))[:100])
