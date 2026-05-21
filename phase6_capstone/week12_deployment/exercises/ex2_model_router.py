"""
Exercise 2: Intelligent Model Router
Goal: Route queries to the cheapest model that can handle them.
Track cost savings vs always using the best model.

pip install litellm python-dotenv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from llm import chat, get_text, calc_cost, MODEL
from dataclasses import dataclass, field

# Local model tiers — swap names to match your Ollama models or cloud models
MODELS = {
    "fast":    {"id": "ollama/qwen2.5:7b"},
    "balanced":{"id": "ollama/mistral"},
    "capable": {"id": "ollama/llama3.2"},
}


@dataclass
class RouterStats:
    calls: int = 0
    routed_cost: float = 0.0
    opus_cost: float = 0.0          # what it would cost if always opus
    model_counts: dict = field(default_factory=lambda: {"haiku": 0, "sonnet": 0, "opus": 0})

    def record(self, model_key: str, input_tok: int, output_tok: int):
        m = MODELS[model_key]
        cost = (input_tok * m["input"] + output_tok * m["output"]) / 1000
        opus = MODELS["opus"]
        opus_cost = (input_tok * opus["input"] + output_tok * opus["output"]) / 1000

        self.calls += 1
        self.routed_cost += cost
        self.opus_cost += opus_cost
        self.model_counts[model_key] += 1

    def savings(self) -> float:
        return self.opus_cost - self.routed_cost

    def savings_pct(self) -> float:
        if self.opus_cost == 0:
            return 0
        return (self.savings() / self.opus_cost) * 100


stats = RouterStats()


def estimate_complexity(query: str) -> int:
    """
    TODO: Classify query complexity as 1 (simple), 2 (moderate), 3 (complex).
    
    Signals for complexity:
    - Word count (short = simple)
    - Keywords: "implement", "debug", "architect", "explain in detail" → complex
    - Question type: factual lookup → simple; reasoning/analysis → complex
    - Code presence → at least moderate
    - Multiple sub-questions → complex
    
    Return 1, 2, or 3.
    """
    word_count = len(query.split())
    query_lower = query.lower()

    complex_keywords = ["implement", "architect", "design", "debug", "analyze",
                        "compare", "tradeoffs", "explain in detail", "step by step"]
    moderate_keywords = ["explain", "how does", "what is the difference", "why"]
    has_code = "```" in query or any(kw in query_lower for kw in ["function", "class", "code"])

    if any(kw in query_lower for kw in complex_keywords) or has_code or word_count > 80:
        return 3
    elif any(kw in query_lower for kw in moderate_keywords) or word_count > 30:
        return 2
    else:
        return 1


def pick_model(query: str) -> str:
    """Return model key (haiku/sonnet/opus) based on complexity."""
    complexity = estimate_complexity(query)
    # TODO: map complexity to model key
    mapping = {1: "haiku", 2: "sonnet", 3: "opus"}
    model_key = mapping[complexity]
    print(f"  [ROUTER] complexity={complexity} → {model_key}")
    return model_key


def routed_call(query: str) -> str:
    """Call the right model for this query, track costs."""
    model_key = pick_model(query)
    model_id = MODELS[model_key]["id"]

    response = chat(
        messages=[{"role": "user", "content": query}],
        model=model_id,
        max_tokens=1024,
    )

    usage = response.usage
    stats.record(model_key, usage.prompt_tokens, usage.completion_tokens)
    return get_text(response)


if __name__ == "__main__":
    test_queries = [
        "What is 2 + 2?",
        "What is the capital of Australia?",
        "Explain how transformers work in machine learning.",
        "What are the tradeoffs between REST and GraphQL APIs?",
        "Implement a binary search tree in Python with insert, delete, and search.",
        "Design a distributed rate limiter that works across multiple servers.",
        "Who was the first US president?",
        "How does backpropagation work?",
    ]

    print("Running routed queries...\n")
    for q in test_queries:
        print(f"Q: {q[:60]}...")
        answer = routed_call(q)
        print(f"A: {answer[:100]}...\n")

    print("=" * 50)
    print(f"ROUTING STATS")
    print(f"Total calls:    {stats.calls}")
    print(f"Model usage:    {stats.model_counts}")
    print(f"Routed cost:    ${stats.routed_cost:.6f}")
    print(f"Opus-only cost: ${stats.opus_cost:.6f}")
    print(f"Savings:        ${stats.savings():.6f} ({stats.savings_pct():.1f}%)")
