"""
Project 37 — Agent Topology Benchmark (solution)
"""
from __future__ import annotations
import asyncio, json, time
from dataclasses import dataclass, field
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv

load_dotenv()
MODEL = "openai/gpt-4o-mini"

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
    max_cost:  float = 0.10
    _steps: int = field(default=0, init=False)
    _cost:  float = field(default=0.0, init=False)

    def tick(self, cost: float = 0.0):
        self._steps += 1
        self._cost  += cost
        if self._steps > self.max_steps:
            raise StepLimitError(f"Exceeded {self.max_steps} steps")
        if self._cost > self.max_cost:
            raise CostCapError(f"Exceeded ${self.max_cost:.3f} cost cap")

    @property
    def steps(self): return self._steps
    @property
    def cost(self): return self._cost


# ── LLM helper ─────────────────────────────────────────────────────────────────
async def llm(messages: list[dict], system: str = "", max_tokens: int = 600) -> tuple[str, float]:
    full = ([{"role": "system", "content": system}] if system else []) + messages
    resp = await litellm.acompletion(model=MODEL, messages=full, max_tokens=max_tokens)
    content = resp.choices[0].message.content or ""
    cost = 0.0
    if resp.usage:
        cost = (resp.usage.prompt_tokens / 1000 * 0.0002) + (resp.usage.completion_tokens / 1000 * 0.0006)
    return content, cost


def mock_web_search(query: str) -> str:
    return (
        "GDPR Article 28 requires a written Data Processing Agreement (DPA) between "
        "a data controller and any processor it engages. The DPA must specify: subject "
        "matter, duration, nature and purpose of processing, type of personal data, "
        "obligations and rights of the controller. Processors must only act on controller "
        "instructions. Violation can result in fines up to €10M or 2% of global turnover."
    )


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
# Topology 1 — Single ReAct
# ══════════════════════════════════════════════════════════════════════════════
class SingleReActAgent:
    name = "Single ReAct"

    async def run(self, task: str) -> TopologyResult:
        guards = AgentGuards(max_steps=8, max_cost=0.05)
        start = time.time()
        total_cost, llm_calls = 0.0, 0
        messages = [{"role": "user", "content": task}]
        system = (
            "You are a research agent. Think step by step. "
            "You may call web_search by outputting: SEARCH: <query>. "
            "When ready, output: ANSWER: <your final answer>."
        )
        try:
            for _ in range(10):
                thought, cost = await llm(messages, system)
                llm_calls += 1
                guards.tick(cost)
                total_cost += cost
                messages.append({"role": "assistant", "content": thought})

                if thought.strip().upper().startswith("SEARCH:"):
                    query = thought.split(":", 1)[1].strip()
                    result = mock_web_search(query)
                    messages.append({"role": "user", "content": f"Search result: {result}"})
                elif thought.strip().upper().startswith("ANSWER:"):
                    answer = thought.split(":", 1)[1].strip()
                    return TopologyResult(self.name, answer, time.time() - start,
                                         total_cost, llm_calls, guards.steps)
                else:
                    # Treat as final answer if no directive
                    return TopologyResult(self.name, thought, time.time() - start,
                                         total_cost, llm_calls, guards.steps)
        except Exception as e:
            return TopologyResult(self.name, "", time.time() - start, total_cost,
                                  llm_calls, guards.steps, error=str(e))
        return TopologyResult(self.name, messages[-1]["content"], time.time() - start,
                              total_cost, llm_calls, guards.steps)


# ══════════════════════════════════════════════════════════════════════════════
# Topology 2 — Orchestrator-Worker
# ══════════════════════════════════════════════════════════════════════════════
class OrchestratorWorkerTopology:
    name = "Orchestrator-Worker"

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost, llm_calls = 0.0, 0
        try:
            # Step 1: Planner
            plan_prompt = (
                f"Decompose this task into exactly 3 JSON subtasks for: researcher, writer, editor.\n"
                f"Task: {task}\n"
                f'Return ONLY JSON: [{{"role":"researcher","subtask":"..."}},'
                f'{{"role":"writer","subtask":"..."}},'
                f'{{"role":"editor","subtask":"..."}}]'
            )
            plan_raw, cost = await llm([{"role": "user", "content": plan_prompt}], max_tokens=300)
            llm_calls += 1; total_cost += cost
            try:
                raw = plan_raw.strip().removeprefix("```json").removesuffix("```").strip()
                subtasks = json.loads(raw)
            except Exception:
                subtasks = [
                    {"role": "researcher", "subtask": f"Research GDPR Article 28 facts for: {task}"},
                    {"role": "writer",     "subtask": f"Write a 200-word draft about GDPR Article 28"},
                    {"role": "editor",     "subtask": f"Edit and polish the GDPR Article 28 explainer"},
                ]

            # Step 2: Workers
            worker_results = []
            specialist_systems = {
                "researcher": "You are a meticulous legal researcher. Return accurate, cited facts.",
                "writer":     "You are a technical writer. Write clear, engaging, well-structured content.",
                "editor":     "You are an editor. Polish prose. Fix errors. Ensure word count ≤220.",
            }
            for sub in subtasks:
                role = sub.get("role", "worker")
                subtask_text = sub.get("subtask", task)
                sys_prompt = specialist_systems.get(role, "You are a specialist agent.")
                result, cost = await llm(
                    [{"role": "user", "content": subtask_text}], system=sys_prompt
                )
                llm_calls += 1; total_cost += cost
                worker_results.append(f"[{role}]: {result}")

            # Step 3: Synthesizer
            combined = "\n\n".join(worker_results)
            synth_prompt = (
                f"Merge these specialist outputs into a single, coherent 200-word answer:\n\n{combined}"
            )
            final, cost = await llm([{"role": "user", "content": synth_prompt}],
                                    system="You are a synthesis agent. Merge outputs cleanly.")
            llm_calls += 1; total_cost += cost

            return TopologyResult(self.name, final, time.time() - start,
                                  total_cost, llm_calls, llm_calls)
        except Exception as e:
            return TopologyResult(self.name, "", time.time() - start, total_cost,
                                  llm_calls, llm_calls, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Topology 3 — Sequential Pipeline
# ══════════════════════════════════════════════════════════════════════════════
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
    name = "Sequential Pipeline"

    async def _parse_json(self, raw: str, model_cls):
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return model_cls.model_validate_json(raw)
        except Exception:
            if model_cls is ResearchOutput:
                return ResearchOutput(facts=[raw[:300]], sources=["web"])
            if model_cls is DraftOutput:
                return DraftOutput(text=raw[:500], word_count=len(raw.split()))
            return FinalOutput(text=raw[:500], changes_made=[])

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost, llm_calls = 0.0, 0
        try:
            # Stage 1: Researcher
            r1_prompt = (
                f"Research GDPR Article 28 for: {task}\n"
                'Return ONLY JSON: {"facts":["..."],"sources":["..."]}'
            )
            r1_raw, cost = await llm([{"role":"user","content":r1_prompt}],
                                     system="You are a legal researcher. Return valid JSON only.")
            llm_calls += 1; total_cost += cost
            research: ResearchOutput = await self._parse_json(r1_raw, ResearchOutput)

            # Stage 2: Writer
            r2_prompt = (
                f"Using these research facts:\n{json.dumps(research.model_dump())}\n\n"
                f"Write a 200-word draft for: {task}\n"
                'Return ONLY JSON: {"text":"...","word_count":<int>}'
            )
            r2_raw, cost = await llm([{"role":"user","content":r2_prompt}],
                                     system="You are a technical writer. Return valid JSON only.")
            llm_calls += 1; total_cost += cost
            draft: DraftOutput = await self._parse_json(r2_raw, DraftOutput)

            # Stage 3: Editor
            r3_prompt = (
                f"Edit and polish this draft (target ≤220 words):\n{draft.text}\n"
                'Return ONLY JSON: {"text":"...","changes_made":["..."]}'
            )
            r3_raw, cost = await llm([{"role":"user","content":r3_prompt}],
                                     system="You are an editor. Return valid JSON only.")
            llm_calls += 1; total_cost += cost
            final: FinalOutput = await self._parse_json(r3_raw, FinalOutput)

            return TopologyResult(self.name, final.text, time.time() - start,
                                  total_cost, llm_calls, 3)
        except Exception as e:
            return TopologyResult(self.name, "", time.time() - start, total_cost,
                                  llm_calls, llm_calls, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Topology 4 — Fan-Out Parallel
# ══════════════════════════════════════════════════════════════════════════════
class FanOutTopology:
    name = "Fan-Out Parallel"
    ASPECTS = [
        ("legal_researcher",  "Research the specific legal requirements of GDPR Article 28."),
        ("example_finder",    "Find a concrete enforcement example or real consequence of GDPR Art 28 non-compliance."),
        ("plain_language",    "Write a plain-language summary of what GDPR Article 28 means for businesses."),
    ]

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost, llm_calls = 0.0, 0
        try:
            # Parallel specialist calls
            calls = [
                llm([{"role": "user", "content": aspect_task}],
                    system=f"You are a {role}. Be concise (≤100 words).")
                for role, aspect_task in self.ASPECTS
            ]
            results = await asyncio.gather(*calls, return_exceptions=True)
            
            aspect_outputs = []
            for (role, _), result in zip(self.ASPECTS, results):
                if isinstance(result, Exception):
                    aspect_outputs.append(f"[{role}]: ERROR")
                else:
                    content, cost = result
                    total_cost += cost
                    llm_calls += 1
                    aspect_outputs.append(f"[{role}]: {content}")

            # Merger
            combined = "\n\n".join(aspect_outputs)
            final, cost = await llm(
                [{"role": "user", "content":
                  f"Merge these into a single coherent 200-word explainer:\n\n{combined}"}],
                system="You are a merger agent. Produce one clean, flowing answer."
            )
            llm_calls += 1; total_cost += cost

            return TopologyResult(self.name, final, time.time() - start,
                                  total_cost, llm_calls, 2)
        except Exception as e:
            return TopologyResult(self.name, "", time.time() - start, total_cost,
                                  llm_calls, llm_calls, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Topology 5 — Debate
# ══════════════════════════════════════════════════════════════════════════════
class DebateTopology:
    name = "Debate"
    max_rounds = 3

    async def run(self, task: str) -> TopologyResult:
        start = time.time()
        total_cost, llm_calls, steps = 0.0, 0, 0
        proposer_sys = "You are a subject-matter expert. Write a high-quality, accurate 200-word answer."
        critic_sys = (
            "You are an adversarial critic. List issues as bullets rated CRITICAL/HIGH/MEDIUM/LOW. "
            "If no CRITICAL or HIGH issues, output exactly: APPROVED"
        )
        try:
            # Initial proposal
            answer, cost = await llm([{"role": "user", "content": task}], system=proposer_sys)
            llm_calls += 1; total_cost += cost; steps += 1

            for round_n in range(self.max_rounds):
                # Critic
                critique, cost = await llm(
                    [{"role": "user", "content": f"Review this answer:\n{answer}"}],
                    system=critic_sys
                )
                llm_calls += 1; total_cost += cost; steps += 1

                if "APPROVED" in critique.upper():
                    break

                # Check for HIGH/CRITICAL issues
                if not any(kw in critique.upper() for kw in ["CRITICAL", "HIGH"]):
                    break  # only low-severity issues — accept

                # Proposer revision
                answer, cost = await llm(
                    [{"role": "user", "content":
                      f"Original task: {task}\n\nYour previous answer:\n{answer}\n\n"
                      f"Critic's feedback:\n{critique}\n\nRevise to address all CRITICAL/HIGH issues."}],
                    system=proposer_sys
                )
                llm_calls += 1; total_cost += cost; steps += 1

            return TopologyResult(self.name, answer, time.time() - start,
                                  total_cost, llm_calls, steps)
        except Exception as e:
            return TopologyResult(self.name, "", time.time() - start, total_cost,
                                  llm_calls, steps, error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class EvalResult:
    score: float
    rationale: str


class Evaluator:
    async def score(self, task: str, answer: str) -> EvalResult:
        prompt = f"Task: {task}\n\nAnswer:\n{answer}\n\n{EVAL_CRITERIA}"
        raw, _ = await llm([{"role": "user", "content": prompt}], max_tokens=100)
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            data = json.loads(raw)
            return EvalResult(score=float(data["score"]), rationale=data.get("rationale", ""))
        except Exception:
            return EvalResult(score=5.0, rationale="parse error")


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
        results = []
        for topo in self.topologies:
            print(f"\nRunning: {topo.name}...")
            result = await topo.run(task)
            if result.answer and not result.error:
                eval_result = await self.evaluator.score(task, result.answer)
                result.quality = eval_result.score
                result.quality_rationale = eval_result.rationale
            results.append(result)
            icon = "✅" if not result.error else "❌"
            print(f"  {icon} quality={result.quality:.1f}/10, "
                  f"cost=${result.cost_usd:.5f}, latency={result.latency_s:.1f}s, "
                  f"calls={result.llm_calls}")
        return results

    def render(self, results: list[TopologyResult]):
        COL = [24, 9, 8, 10, 8, 7]
        def row(*cells):
            return " │ ".join(str(c).ljust(w) for c, w in zip(cells, COL))

        sep = "─" * (sum(COL) + 3 * len(COL))
        print("\n" + "═" * (sum(COL) + 3 * len(COL)))
        print("  AGENT TOPOLOGY BENCHMARK — Decision Matrix")
        print("═" * (sum(COL) + 3 * len(COL)))
        print(row("Topology", "Quality", "Cost $", "Latency s", "LLM Calls", "Steps"))
        print(sep)
        for r in results:
            if r.error:
                print(row(r.topology, "ERROR", "—", "—", "—", "—") + f"  ← {r.error[:30]}")
            else:
                print(row(r.topology, f"{r.quality:.1f}/10",
                          f"${r.cost_usd:.4f}", f"{r.latency_s:.1f}",
                          str(r.llm_calls), str(r.steps)))
        print(sep)

        good = [r for r in results if not r.error]
        if good:
            fastest  = min(good, key=lambda r: r.latency_s)
            cheapest = min(good, key=lambda r: r.cost_usd)
            highest  = max(good, key=lambda r: r.quality)
            # Best quality/cost ratio
            def qc(r): return r.quality / max(r.cost_usd * 1000, 0.0001)
            best_ratio = max(good, key=qc)

            print("\nRecommendations:")
            print(f"  Fastest:           {fastest.topology:<24} ({fastest.latency_s:.1f}s, quality {fastest.quality:.1f})")
            print(f"  Cheapest:          {cheapest.topology:<24} (${cheapest.cost_usd:.4f}, quality {cheapest.quality:.1f})")
            print(f"  Highest quality:   {highest.topology:<24} ({highest.quality:.1f}/10, ${highest.cost_usd:.4f})")
            print(f"  Best quality/cost: {best_ratio.topology:<24} ({best_ratio.quality:.1f}/10 per 0.1¢)")

            print("\nWhen to use each:")
            print("  Single ReAct       → Simple tasks, tight latency, cost-sensitive")
            print("  Orchestrator-Worker → Complex tasks with distinct specializations")
            print("  Sequential Pipeline → Quality gates required between stages")
            print("  Fan-Out Parallel    → Independent subtasks, moderate latency budget")
            print("  Debate             → High-stakes decisions where quality >> cost")


async def main():
    print("=" * 65)
    print("Project 37 — Agent Topology Benchmark (Solution)")
    print("=" * 65)
    print(f"\nTask: {BENCHMARK_TASK[:80]}...")

    runner = BenchmarkRunner()
    results = await runner.run(BENCHMARK_TASK)
    runner.render(results)

    with open("benchmark_results.json", "w") as f:
        json.dump([{"topology": r.topology, "quality": r.quality, "cost": r.cost_usd,
                    "latency_s": r.latency_s, "llm_calls": r.llm_calls, "steps": r.steps,
                    "error": r.error, "rationale": r.quality_rationale}
                   for r in results], f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
