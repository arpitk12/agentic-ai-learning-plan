"""
SOLUTION — Exercise 2: Budget-Limited Agent with Cost Tracking
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import structlog
from dotenv import load_dotenv
from dataclasses import dataclass
from llm import chat, get_text, calc_cost, MODEL

load_dotenv()
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

        response = chat(messages, model=model, max_tokens=512)
        tracker.record(response.usage.prompt_tokens, response.usage.completion_tokens)
        reply = get_text(response)
        messages.append({"role": "assistant", "content": reply})

        if stop_reason_val := response.choices[0].finish_reason:
            if stop_reason_val == "stop":
                log.info("agent_complete", **tracker.summary())
                return reply

    return "[max steps reached]"


if __name__ == "__main__":
    result = budget_agent(
        "Explain the differences between supervised and unsupervised learning in 3 paragraphs.",
        budget_usd=0.01,
    )
    print(result)
