"""
SOLUTION — Exercise 2: Intelligent Model Router
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import re
from dotenv import load_dotenv
from dataclasses import dataclass, field
from llm import chat, get_text, calc_cost

load_dotenv()

# Map friendly names to LiteLLM model strings — update to match your provider
MODELS = {
    "haiku":  "groq/llama-3.1-8b-instant",
    "sonnet": "groq/llama-3.3-70b-versatile",
    "opus":   "groq/llama-3.3-70b-versatile",
}


@dataclass
class RouterStats:
    calls: int = 0
    routed_cost: float = 0.0
    opus_cost: float = 0.0
    model_counts: dict = field(default_factory=lambda: {"haiku": 0, "sonnet": 0, "opus": 0})

    def record(self, model_key: str, model_id: str, input_tok: int, output_tok: int):
        self.routed_cost += calc_cost(model_id, input_tok, output_tok)
        self.opus_cost += calc_cost(MODELS["opus"], input_tok, output_tok)
        self.calls += 1
        self.model_counts[model_key] += 1

    def savings(self) -> float:
        return self.opus_cost - self.routed_cost

    def savings_pct(self) -> float:
        return (self.savings() / self.opus_cost * 100) if self.opus_cost else 0


stats = RouterStats()

COMPLEX_KEYWORDS = {
    "implement", "design", "architect", "debug", "refactor", "optimize",
    "analyze", "compare", "evaluate", "critique", "explain in detail",
    "write a", "create a", "build a", "step by step", "comprehensive",
}
SIMPLE_PATTERNS = [
    r"^what is \w+\??\s*$",
    r"^who (is|was) \w+\??\s*$",
    r"^when (did|was|is) .{0,40}\??\s*$",
    r"^\d[\d\s+\-*/()]+\??\s*$",
    r"^(yes or no|true or false)",
]


def estimate_complexity(query: str) -> int:
    q = query.lower().strip()
    words = q.split()

    # Simple: short factual questions
    for pat in SIMPLE_PATTERNS:
        if re.match(pat, q):
            return 1
    if len(words) <= 8 and "?" in query:
        return 1

    # Complex: code, architecture, detailed analysis
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return 3
    if "```" in query or len(words) > 80:
        return 3

    # Moderate: everything else
    return 2


def pick_model(query: str) -> str:
    complexity = estimate_complexity(query)
    return {1: "haiku", 2: "sonnet", 3: "opus"}[complexity]


def routed_call(query: str) -> tuple[str, str]:
    """Returns (response_text, model_key_used)."""
    model_key = pick_model(query)
    model_id = MODELS[model_key]

    response = chat([{"role": "user", "content": query}], model=model_id, max_tokens=512)
    stats.record(model_key, model_id, response.usage.prompt_tokens, response.usage.completion_tokens)
    return get_text(response), model_key


if __name__ == "__main__":
    test_queries = [
        "What is the capital of France?",
        "Explain the difference between supervised and unsupervised learning.",
        "Design and implement a production-ready async job queue system in Python with Redis, including retry logic, dead letter queues, and monitoring.",
        "What year was Python created?",
        "Write a comprehensive analysis of microservices vs monolithic architecture, including tradeoffs, migration strategies, and when to use each.",
        "2 + 2 = ?",
    ]

    print("=== Model Router Demo ===\n")
    for query in test_queries:
        response, model = routed_call(query)
        complexity = estimate_complexity(query)
        print(f"Q: {query[:60]}...")
        print(f"   Model: {model} (complexity={complexity})")
        print(f"   A: {response[:80]}...\n")

    print("=== Stats ===")
    print(f"Total calls:    {stats.calls}")
    print(f"Model usage:    {stats.model_counts}")
    print(f"Routed cost:    ${stats.routed_cost:.6f}")
    print(f"Opus-only cost: ${stats.opus_cost:.6f}")
    print(f"Savings:        ${stats.savings():.6f} ({stats.savings_pct():.1f}%)")
