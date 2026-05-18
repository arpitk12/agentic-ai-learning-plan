"""
Exercise 2: Budget-Limited Agent with Cost Tracking
Goal: Track token usage and cost per run. Stop if budget exceeded.

pip install litellm python-dotenv structlog
"""
import structlog
from llm import chat, get_text, calc_cost, MODEL
from dataclasses import dataclass, field

log = structlog.get_logger()

DEFAULT_MODEL = MODEL


@dataclass
class CostTracker:
    model: str
    budget_usd: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    steps: int = 0

    def record(self, input_tokens: int, output_tokens: int):
        cost = calc_cost(self.model, input_tokens, output_tokens)
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
            "under_budget": not self.is_over_budget()
        }


def budget_agent(query: str, budget_usd: float = 0.01, model: str = DEFAULT_MODEL) -> str:
    """
    TODO: Implement a ReAct agent that:
    1. Creates a CostTracker with the given budget
    2. After each API call, records tokens with tracker.record()
    3. Before each step, checks tracker.is_over_budget()
    4. If over budget, returns partial result with warning
    5. Logs tracker.summary() at the end
    """
    tracker = CostTracker(model=model, budget_usd=budget_usd)
    messages = [{"role": "user", "content": query}]

    for step in range(10):
        if tracker.is_over_budget():
            log.warning("budget_exceeded", **tracker.summary())
            return f"[Budget ${budget_usd} exceeded after {tracker.steps} steps] Partial answer unavailable."

        # TODO: Call the API
        # TODO: Record token usage with tracker.record()
        # TODO: Handle end_turn and tool_use stop reasons
        break

    log.info("agent_complete", **tracker.summary())
    return "Not implemented"


if __name__ == "__main__":
    result = budget_agent(
        "Explain the history of the Roman Empire in great detail",
        budget_usd=0.002  # very tight — should cut off early
    )
    print(result)
