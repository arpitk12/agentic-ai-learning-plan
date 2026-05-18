"""
SOLUTION — Exercise 2: Budget-Limited Agent with Cost Tracking
"""
import structlog
from anthropic import Anthropic
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
client = Anthropic()
log = structlog.get_logger()

COST_PER_1K = {
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-6":         {"input": 0.003,   "output": 0.015},
    "claude-opus-4-6":           {"input": 0.015,   "output": 0.075},
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class CostTracker:
    model: str
    budget_usd: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    steps: int = 0

    def record(self, input_tokens: int, output_tokens: int):
        rates = COST_PER_1K.get(self.model, COST_PER_1K[DEFAULT_MODEL])
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.steps += 1
        log.info("llm_call",
                 step=self.steps,
                 input_tokens=input_tokens,
                 output_tokens=output_tokens,
                 step_cost_usd=round(cost, 6),
                 total_cost_usd=round(self.total_cost_usd, 6),
                 budget_remaining=round(self.budget_usd - self.total_cost_usd, 6))

    def is_over_budget(self) -> bool:
        return self.total_cost_usd >= self.budget_usd

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": self.budget_usd,
            "under_budget": not self.is_over_budget(),
        }


def budget_agent(query: str, budget_usd: float = 0.005, model: str = DEFAULT_MODEL) -> str:
    tracker = CostTracker(model=model, budget_usd=budget_usd)
    messages = [{"role": "user", "content": query}]

    log.info("agent_start", query=query, budget_usd=budget_usd, model=model)

    for _step in range(20):
        if tracker.is_over_budget():
            log.warning("budget_exceeded", **tracker.summary())
            return f"[BUDGET EXCEEDED after {tracker.steps} steps. Partial answer not available.]"

        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=messages,
        )
        tracker.record(response.usage.input_tokens, response.usage.output_tokens)
        messages.append({"role": "assistant", "content": response.content[0].text})

        if response.stop_reason == "end_turn":
            log.info("agent_complete", **tracker.summary())
            return response.content[0].text

    return "[max steps reached]"


if __name__ == "__main__":
    result = budget_agent(
        "Explain the differences between supervised and unsupervised learning in 3 paragraphs.",
        budget_usd=0.01,
    )
    print(result)
