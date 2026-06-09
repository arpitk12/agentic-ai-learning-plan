"""
Project 37 — Agent Topology Benchmark (starter)
================================================
Benchmark 5 agent topologies on the same task.
Produce a decision matrix: quality / cost / latency / LLM calls / steps.

Companion: guide/13_system_design.md §2 — Agent Topology Patterns

Fill in every # TODO block. Do NOT look at solution/ until you've tried.
"""
from __future__ import annotations
import asyncio, time, json
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()
MODEL = "openai/gpt-4o-mini"

# ── Shared benchmark task ──────────────────────────────────────────────────────
BENCHMARK_TASK = (
    "Research and write a concise 200-word explainer on GDPR Article 28 "
    "(Data Processing Agreements). Include: what it requires, who it applies to, "
    "and one real consequence of non-compliance. Use plain language."
)

EVAL_CRITERIA = (
    "Score 0–10 on: accuracy (GDPR Article 28 facts), completeness (all 3 required "
    "elements), conciseness (≤220 words), structure (readable flow). "
    "Return ONLY valid JSON: {\"score\": <int>, \"rationale\": \"<1 sentence>\"}"
)

# ── Guards ─────────────────────────────────────────────────────────────────────
class StepLimitError(Exception): pass
class CostCapError(Exception): pass

@dataclass
class AgentGuards:
    max_steps: int = 10
    max_cost: float = 0.10
    _steps: int = field(default=0, init=False)
    _cost: float = field(default=0.0, init=False)

    def tick(self, cost: float = 0.0):
        """Call after each LLM call. Raises if limits exceeded."""
        # TODO 1: increment _steps and _cost; raise StepLimitError or CostCapError if over limit
        raise NotImplementedError

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def cost(self) -> float:
        return self._cost


# ── LLM helper ────────────────────────────────────────────────────────────────
async def llm(messages: list[dict], system: str = "", max_tokens: int = 600) -> tuple[str, float]:
    """Returns (content, cost_usd). cost_usd = 0 if usage unavailable."""
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    resp = await litellm.acompletion(model=MODEL, messages=full_messages, max_tokens=max_tokens)
    content = resp.choices[0].message.content or ""
    cost = 0.0
    if resp.usage:
        in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
        cost = (in_tok / 1000 * 0.0002) + (out_tok / 1000 * 0.0006)
    return content, cost


def mock_web_search(query: str) -> str:
    """Simulated web search — returns canned GDPR Article 28 facts."""
    return (
        "GDPR Article 28 requires a written Data Processing Agreement (DPA) between "
        "a data controller and any processor it engages. The DPA must specify: "
        "subject matter, duration, nature and purpose of processing, type of personal data, "
        "obligations and rights of the controller. Processors must only act on controller "
        "instructions. Violation can result in fines up to €10M or 2% of global turnover."
    )


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class TopologyResult:
    topology: str
    answer: str
    latency_s: float
    cost_usd: float
    llm_calls: int
    steps: int
    error: str = ""
    quality: float = 0.0
    quality_rationale: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY 1 — Single ReAct Agent
# ══════════════════════════════════════════════════════════════════════════════
class SingleReActAgent:
    """
    Standard ReAct loop: think → (optional) tool call → observe → repeat → final answer.
    Tools available: web_search(query) → str
    """
    name = "Single ReAct"

    async def run(self, task: str) -> TopologyResult:
        guards = AgentGuards(max_steps=8, max_cost=0.05)
        start = time.time()
        total_cost = 0.0
        llm_calls = 0
        messages = [{"role": "user", "content": task}]
        system = (
            "You are a research agent. Think step by step. "
            "You may call web_search(query) by outputting: SEARCH: <query>. "
            "When you have enough information, output: ANSWER: <your final answer>."
        )

        # TODO 2: Implement the ReAct loop
        # - Call llm(messages, system) → get thought
        # - If thought starts with "SEARCH:", call mock_web_search and append result to messages
        # - If thought starts with "ANSWER:", extract and return the answer
        # - Call guards.tick(cost) after each LLM call
        # - Limit: max 8 iterations (guards will raise)
        # - On exception: return TopologyResult with error field set
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY 2 — Orchestrator + Workers
# ══════════════════════════════════════════════════════════════════════════════
class OrchestratorWorkerTopology:
    """
    Planner decomposes task into subtasks (JSON list).
    Each subtask runs with a specialist prompt.
    Synthesizer merges results.
    """
    name = "Orchestrator-Worker"

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost = 0.0
        llm_calls = 0

        # TODO 3: Step 1 — Planner call
        # Prompt: ask for JSON list of 3 subtasks (researcher / writer / editor focus)
        # Parse the JSON. If parse fails, use 3 default subtasks.

        # TODO 4: Step 2 — Workers (sequential in this version)
        # For each subtask, call llm() with an appropriate specialist system prompt.
        # Collect results as a list of strings.

        # TODO 5: Step 3 — Synthesizer
        # Pass all worker results to a final LLM call. Ask it to merge into a single answer.

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY 3 — Sequential Pipeline (Pydantic typed handoffs)
# ══════════════════════════════════════════════════════════════════════════════
from pydantic import BaseModel

class ResearchOutput(BaseModel):
    facts: list[str]
    sources: list[str]

class DraftOutput(BaseModel):
    text: str
    word_count: int

class FinalOutput(BaseModel):
    text: str
    changes_made: list[str]


class SequentialPipelineTopology:
    """
    3 fixed stages with structured Pydantic handoffs.
    Stage 1: Researcher → ResearchOutput
    Stage 2: Writer → DraftOutput
    Stage 3: Editor → FinalOutput
    """
    name = "Sequential Pipeline"

    async def _parse_json(self, raw: str, model_cls: type[BaseModel]) -> BaseModel:
        """Strip markdown fences and parse JSON into model_cls. Return default on failure."""
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return model_cls.model_validate_json(raw)
        except Exception:
            # Return a safe default so the pipeline continues
            if model_cls is ResearchOutput:
                return ResearchOutput(facts=[raw[:300]], sources=["web"])
            if model_cls is DraftOutput:
                return DraftOutput(text=raw[:500], word_count=len(raw.split()))
            return FinalOutput(text=raw[:500], changes_made=[])

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost = 0.0
        llm_calls = 0

        # TODO 6: Stage 1 — Researcher
        # System: "Research agent. Return JSON matching: {facts: [...], sources: [...]}"
        # Parse response into ResearchOutput using _parse_json()

        # TODO 7: Stage 2 — Writer
        # Pass ResearchOutput as context. Return JSON matching DraftOutput schema.

        # TODO 8: Stage 3 — Editor
        # Pass DraftOutput as context. Return JSON matching FinalOutput schema.
        # Final answer is FinalOutput.text

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY 4 — Fan-Out Parallel
# ══════════════════════════════════════════════════════════════════════════════
class FanOutTopology:
    """
    Split task into 3 aspects. Run all 3 agents in parallel. Merge.
    """
    name = "Fan-Out Parallel"

    ASPECTS = [
        ("legal_researcher",  "Research the specific legal requirements of GDPR Article 28."),
        ("example_finder",    "Find a concrete real-world example of GDPR Article 28 enforcement."),
        ("plain_language",    "Write a plain-language summary of what GDPR Article 28 means for businesses."),
    ]

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost = 0.0
        llm_calls = 0

        # TODO 9: Launch all 3 specialist agents in parallel using asyncio.gather()
        # Each call: llm([{"role":"user","content": aspect_task}], system=specialist_name)
        # Collect (content, cost) pairs

        # TODO 10: Merger LLM call
        # Combine all 3 outputs into a single coherent 200-word answer

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY 5 — Debate (Adversarial Review)
# ══════════════════════════════════════════════════════════════════════════════
class DebateTopology:
    """
    Proposer writes initial answer.
    Critic rates issues as CRITICAL / HIGH / MEDIUM / LOW.
    Proposer revises until no CRITICAL/HIGH or max_rounds reached.
    """
    name = "Debate"
    max_rounds: int = 3

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost = 0.0
        llm_calls = 0

        # TODO 11: Initial proposal
        # Proposer system: "You are a subject-matter expert. Write a high-quality answer."

        # TODO 12: Debate loop (max self.max_rounds)
        # Critic system: "Review the answer. List issues as CRITICAL/HIGH/MEDIUM/LOW bullets.
        #                 If no CRITICAL or HIGH issues, output: APPROVED"
        # If critic outputs "APPROVED" → break
        # Else: pass critique to Proposer for revision with: "Revise to fix: <critique>"

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class EvalResult:
    score: float
    rationale: str


class Evaluator:
    async def score(self, task: str, answer: str) -> EvalResult:
        # TODO 13: LLM-as-judge call
        # Ask model to evaluate answer against EVAL_CRITERIA (defined at top of file)
        # Parse JSON response → EvalResult
        # On parse error: return EvalResult(score=5.0, rationale="parse error")
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════
class BenchmarkRunner:
    def __init__(self):
        self.evaluator = Evaluator()
        self.topologies = [
            SingleReActAgent(),
            OrchestratorWorkerTopology(),
            SequentialPipelineTopology(),
            FanOutTopology(),
            DebateTopology(),
        ]

    async def run(self, task: str) -> list[TopologyResult]:
        results: list[TopologyResult] = []
        for topo in self.topologies:
            print(f"\nRunning: {topo.name}...")
            try:
                result = await topo.run(task)
                if not result.error:
                    eval_result = await self.evaluator.score(task, result.answer)
                    result.quality = eval_result.score
                    result.quality_rationale = eval_result.rationale
                results.append(result)
                print(f"  ✅ done — quality={result.quality}/10, cost=${result.cost_usd:.5f}, "
                      f"latency={result.latency_s:.1f}s, calls={result.llm_calls}")
            except Exception as e:
                results.append(TopologyResult(
                    topology=topo.name, answer="", latency_s=0, cost_usd=0,
                    llm_calls=0, steps=0, error=str(e)
                ))
                print(f"  ❌ {e}")
        return results

    def render(self, results: list[TopologyResult]):
        # TODO 14: Print a formatted table using str.ljust() or the rich library
        # Columns: Topology | Quality/10 | Cost $ | Latency s | LLM Calls | Steps
        # After the table: print Recommendations (best quality/cost, fastest, highest quality)
        raise NotImplementedError


async def main():
    print("=" * 65)
    print("Project 37 — Agent Topology Benchmark")
    print("=" * 65)
    print(f"\nTask: {BENCHMARK_TASK[:80]}...")

    runner = BenchmarkRunner()
    results = await runner.run(BENCHMARK_TASK)
    print()
    runner.render(results)

    # Save raw results for analysis
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump([
            {"topology": r.topology, "quality": r.quality, "cost": r.cost_usd,
             "latency_s": r.latency_s, "llm_calls": r.llm_calls, "steps": r.steps,
             "error": r.error, "rationale": r.quality_rationale}
            for r in results
        ], f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
