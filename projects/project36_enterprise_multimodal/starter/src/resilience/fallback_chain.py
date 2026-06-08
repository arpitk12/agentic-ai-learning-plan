"""
src/resilience/fallback_chain.py — Ordered model fallback with circuit breakers.

TODOs:
  1. Implement FallbackChain — tries models in order, uses circuit breakers
     to skip tripped models, retries with exponential backoff
"""
from __future__ import annotations
import asyncio
import time


# ── TODO 1: FallbackChain class ───────────────────────────────────────────────
# class FallbackChain:
#     """
#     Tries each model in `models` list in order.
#     Each model has its own CircuitBreaker.
#     If all models fail (or are tripped), raises RuntimeError.
#     """
#     def __init__(self, models: list[str], failure_threshold: int = 3,
#                  recovery_timeout: float = 30.0, max_retries: int = 2):
#         from src.resilience.circuit_breaker import CircuitBreaker
#         self._models = models
#         self._breakers = {m: CircuitBreaker(m, failure_threshold, recovery_timeout) for m in models}
#         self._max_retries = max_retries
#
#     async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
#         """
#         Try each model in order.
#
#         For each model:
#           1. Skip if breaker.can_attempt() is False (OPEN circuit)
#           2. Try with exponential backoff retry (max_retries):
#              - await litellm.acompletion(model=model, messages=messages, **kwargs)
#              - On success: breaker.record_success(), return (reply, model)
#              - On exception: breaker.record_failure(), continue to next retry
#           3. If all retries fail for this model: move to next model
#
#         Raises RuntimeError if all models fail.
#
#         Returns:
#             tuple[str, str] — (reply_text, model_name_that_succeeded)
#         """
#         import litellm
#         for model in self._models:
#             breaker = self._breakers[model]
#             if not breaker.can_attempt():
#                 continue
#             for attempt in range(self._max_retries + 1):
#                 try:
#                     resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
#                     breaker.record_success()
#                     return resp.choices[0].message.content, model
#                 except Exception as e:
#                     breaker.record_failure()
#                     if attempt < self._max_retries:
#                         await asyncio.sleep(2 ** attempt)   # exponential backoff
#         raise RuntimeError("All models in fallback chain failed")
#
#     def status(self) -> dict[str, str]:
#         """Return {model: state} for all models in chain."""
#         return {m: b.state.value for m, b in self._breakers.items()}

raise NotImplementedError("Implement FallbackChain in src/resilience/fallback_chain.py")
