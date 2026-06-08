"""
src/guardrails/safety_checker.py — Layer 3: LlamaGuard / GPT-4o safety classifier.

TODO:
  1. implement load_llama_guard() — load transformers model (GPU required)
  2. implement check_safety() — call LlamaGuard if loaded, else GPT-4o-mini fallback
"""
from __future__ import annotations
import asyncio

# Module-level cache for the LlamaGuard model (None if not loaded)
_LLAMA_GUARD_MODEL = None
_LLAMA_GUARD_TOKENIZER = None


# ── TODO 1: Load LlamaGuard (optional — needs GPU) ───────────────────────────
def load_llama_guard(model_id: str = "meta-llama/Llama-Guard-3-8B") -> bool:
    """
    Load the LlamaGuard model into the module-level cache.

    Steps:
      1a. from transformers import AutoTokenizer, AutoModelForCausalLM
      1b. _LLAMA_GUARD_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
      1c. _LLAMA_GUARD_MODEL = AutoModelForCausalLM.from_pretrained(
              model_id, torch_dtype=torch.bfloat16, device_map="auto")
      1d. Return True
      1e. On ImportError or RuntimeError (no GPU): return False

    Note: This function is optional — check_safety falls back to GPT-4o-mini
          when _LLAMA_GUARD_MODEL is None.
    """
    global _LLAMA_GUARD_MODEL, _LLAMA_GUARD_TOKENIZER
    # try:
    #     import torch
    #     from transformers import AutoTokenizer, AutoModelForCausalLM
    #     ...
    # except Exception:
    #     return False
    raise NotImplementedError


# ── TODO 2: Check safety ──────────────────────────────────────────────────────
async def check_safety(text: str, model: str = "openai/gpt-4o-mini") -> tuple[bool, str]:
    """
    Classify whether `text` is safe using LlamaGuard or GPT-4o-mini fallback.

    Steps:
      2a. If _LLAMA_GUARD_MODEL is not None:
              tokenize → generate → decode output
              LlamaGuard outputs "safe" or "unsafe\nS1" (with violation category)
              Return (True, "") if "safe", (False, category) if "unsafe"
      2b. Else (fallback to GPT-4o-mini):
              import litellm
              Prompt: "Is the following text safe for a compliance AI assistant?
                       Text: {text}
                       Return JSON: {"safe": bool, "reason": str, "category": str}"
              Parse JSON response
              Return (result["safe"], result.get("reason", ""))
      2c. On any exception: return (True, "") — fail open (don't block on error)

    Returns:
        tuple[bool, str] — (is_safe, reason/category string)
    """
    raise NotImplementedError
