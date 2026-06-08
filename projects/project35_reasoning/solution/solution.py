"""
Project 35 SOLUTION — Advanced Reasoning Agent
Tree of Thought + o3-mini (high reasoning_effort) + Monte Carlo Tree Search.
"""
from __future__ import annotations
import os, asyncio, json, math, random
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()


# ── 1. Tree of Thought (Breadth-First Search) ─────────────────────────────────

@dataclass
class ThoughtNode:
    thought: str
    depth: int
    score: float = 0.0
    children: list["ThoughtNode"] = field(default_factory=list)
    parent: "ThoughtNode | None" = None

    def path(self) -> list[str]:
        """Return the chain of thoughts from root to this node."""
        node, chain = self, []
        while node:
            chain.append(node.thought)
            node = node.parent
        return list(reversed(chain))


async def generate_thoughts(
    problem: str, current_thought: str, n: int = 3
) -> list[str]:
    """Generate n candidate next thoughts for the current reasoning step."""
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Current reasoning: {current_thought}

Generate {n} DIFFERENT plausible next reasoning steps. Be creative and explore different angles.
Return JSON array of {n} strings: ["step 1", "step 2", "step 3"]""",
        }],
        response_format={"type": "json_object"},
        temperature=0.8,  # higher temp for diverse branches
    )
    data = json.loads(resp.choices[0].message.content)
    # Handle both {"thoughts": [...]} and direct array responses
    if isinstance(data, list):
        return data[:n]
    for key in ("thoughts", "steps", "options", "reasoning_steps"):
        if key in data and isinstance(data[key], list):
            return data[key][:n]
    return list(data.values())[0][:n] if data else ["Continue reasoning..."]


async def score_thought(problem: str, thought_chain: list[str]) -> float:
    """Score a chain of thoughts from 0.0 (terrible) to 1.0 (excellent)."""
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Reasoning chain:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(thought_chain))}

Rate this reasoning chain 1-10 on: logical coherence, correctness, completeness.
Return JSON: {{"score": <integer 1-10>, "reason": "<brief reason>"}}""",
        }],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    data = json.loads(resp.choices[0].message.content)
    raw = float(data.get("score", 5))
    return raw / 10.0


async def tree_of_thought_bfs(
    problem: str,
    branching_factor: int = 3,
    max_depth: int = 3,
    beam_width: int = 2,   # keep top-k at each level
) -> dict:
    """
    BFS Tree of Thought:
    - At each depth, expand the best `beam_width` nodes.
    - Each node generates `branching_factor` children.
    - Score all children and keep the top `beam_width` for the next level.
    """
    print(f"  ToT BFS: depth={max_depth}, branching={branching_factor}, beam={beam_width}")

    # Root node — initial problem statement
    root = ThoughtNode(thought=f"Let me analyze: {problem}", depth=0, score=0.5)
    beam = [root]

    for depth in range(1, max_depth + 1):
        candidates: list[ThoughtNode] = []

        # Expand all nodes in current beam
        expand_tasks = []
        for node in beam:
            expand_tasks.append(generate_thoughts(problem, node.thought, branching_factor))

        all_thoughts_lists = await asyncio.gather(*expand_tasks)

        # Create child nodes
        new_nodes: list[ThoughtNode] = []
        for node, thoughts in zip(beam, all_thoughts_lists):
            for t in thoughts:
                child = ThoughtNode(thought=t, depth=depth, parent=node)
                node.children.append(child)
                new_nodes.append(child)

        # Score all new nodes in parallel
        score_tasks = [score_thought(problem, n.path()) for n in new_nodes]
        scores = await asyncio.gather(*score_tasks)
        for node, score in zip(new_nodes, scores):
            node.score = score

        # Keep top beam_width nodes
        beam = sorted(new_nodes, key=lambda n: n.score, reverse=True)[:beam_width]
        best = beam[0]
        print(f"    Depth {depth}: best_score={best.score:.2f} | thought='{best.thought[:60]}...'")

    # Best leaf node
    best_node = beam[0]
    best_path = best_node.path()

    # Synthesise final answer from best thought chain
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Best reasoning chain found:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(best_path))}

Based on this reasoning, provide a clear, concise final answer.""",
        }],
        temperature=0.0,
    )
    return {
        "method": "tree_of_thought_bfs",
        "best_score": best_node.score,
        "reasoning_chain": best_path,
        "final_answer": resp.choices[0].message.content,
        "nodes_explored": sum(branching_factor ** d for d in range(max_depth + 1)),
    }


# ── 2. o3-mini with Extended Thinking ─────────────────────────────────────────

async def o3_mini_reasoning(problem: str, reasoning_effort: str = "high") -> dict:
    """
    Use o3-mini with high reasoning_effort for complex multi-step problems.
    Falls back to chain-of-thought prompt if o3-mini is unavailable.
    """
    try:
        resp = await litellm.acompletion(
            model="openai/o3-mini",
            messages=[{"role": "user", "content": problem}],
            reasoning_effort=reasoning_effort,  # "low" | "medium" | "high"
        )
        return {
            "method": "o3_mini",
            "reasoning_effort": reasoning_effort,
            "answer": resp.choices[0].message.content,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    except Exception as e:
        print(f"  o3-mini unavailable ({e}), falling back to GPT-4o chain-of-thought")
        resp = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are an expert reasoning assistant. Think step-by-step before answering.",
            }, {
                "role": "user",
                "content": f"<think>\nWork through this carefully step by step.\n</think>\n\n{problem}",
            }],
            temperature=0.2,
        )
        return {
            "method": "gpt4o_chain_of_thought_fallback",
            "reasoning_effort": reasoning_effort,
            "answer": resp.choices[0].message.content,
        }


# ── 3. Monte Carlo Tree Search (MCTS) ─────────────────────────────────────────

@dataclass
class MCTSNode:
    thought: str
    depth: int
    visits: int = 0
    total_score: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)
    parent: "MCTSNode | None" = None
    _untried: list[str] | None = None

    @property
    def ucb1(self) -> float:
        """UCB1 score: exploitation + exploration."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_score / self.visits
        exploration_c = math.sqrt(2)
        parent_visits = self.parent.visits if self.parent else self.visits
        exploration = exploration_c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def path(self) -> list[str]:
        node, chain = self, []
        while node:
            chain.append(node.thought)
            node = node.parent
        return list(reversed(chain))


async def mcts_reasoning(
    problem: str,
    simulations: int = 20,
    branching_factor: int = 3,
    max_depth: int = 3,
) -> dict:
    """
    Monte Carlo Tree Search for reasoning:
    1. Selection: pick node with best UCB1
    2. Expansion: generate children if not expanded
    3. Simulation: rollout to leaf with random thoughts
    4. Backpropagation: update visit counts and scores up the tree
    """
    print(f"  MCTS: simulations={simulations}, branching={branching_factor}, depth={max_depth}")

    root = MCTSNode(thought=f"Initial analysis of: {problem}", depth=0)

    def select(node: MCTSNode) -> MCTSNode:
        """Descend using UCB1 until an unexpanded node or leaf."""
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1)
        return node

    async def expand(node: MCTSNode) -> MCTSNode:
        """Expand node by generating children."""
        if node.depth >= max_depth:
            return node
        thoughts = await generate_thoughts(problem, node.thought, branching_factor)
        for t in thoughts:
            child = MCTSNode(thought=t, depth=node.depth + 1, parent=node)
            node.children.append(child)
        return random.choice(node.children) if node.children else node

    async def simulate(node: MCTSNode) -> float:
        """Score the thought path from root to this node."""
        return await score_thought(problem, node.path())

    def backpropagate(node: MCTSNode, score: float):
        """Propagate score up through ancestors."""
        while node:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # Run MCTS simulations
    for sim in range(simulations):
        leaf = select(root)
        if leaf.visits > 0 and leaf.depth < max_depth:
            leaf = await expand(leaf)
        score = await simulate(leaf)
        backpropagate(leaf, score)
        if (sim + 1) % 5 == 0:
            best = max(root.children, key=lambda n: n.total_score / max(n.visits, 1), default=root)
            print(f"    Sim {sim+1}/{simulations}: best_avg={best.total_score/max(best.visits,1):.2f}")

    # Best child of root
    if not root.children:
        best_child = root
    else:
        best_child = max(root.children, key=lambda n: n.total_score / max(n.visits, 1))

    # Synthesise answer
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Best MCTS reasoning path found (avg score {best_child.total_score/max(best_child.visits,1):.2f}):
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(best_child.path()))}

Provide the final answer based on this reasoning.""",
        }],
        temperature=0.0,
    )

    return {
        "method": "mcts",
        "simulations": simulations,
        "best_avg_score": best_child.total_score / max(best_child.visits, 1),
        "best_path": best_child.path(),
        "final_answer": resp.choices[0].message.content,
    }


# ── 4. Method Comparison ──────────────────────────────────────────────────────

async def compare_reasoning_methods(problem: str) -> dict:
    """Run all 3 reasoning methods and compare results."""
    print(f"\nComparing reasoning methods on:\n  '{problem[:80]}...'\n")

    # Run methods sequentially to avoid rate limits
    print("Running Tree of Thought BFS...")
    tot = await tree_of_thought_bfs(problem, branching_factor=3, max_depth=2, beam_width=2)

    print("\nRunning o3-mini (extended thinking)...")
    o3 = await o3_mini_reasoning(problem, reasoning_effort="high")

    print("\nRunning MCTS (10 simulations)...")
    mcts = await mcts_reasoning(problem, simulations=10, branching_factor=2, max_depth=2)

    return {
        "problem": problem,
        "tree_of_thought": {
            "score": tot["best_score"],
            "nodes_explored": tot["nodes_explored"],
            "answer_preview": tot["final_answer"][:200],
        },
        "o3_mini": {
            "method": o3["method"],
            "answer_preview": o3["answer"][:200],
        },
        "mcts": {
            "avg_score": mcts["best_avg_score"],
            "simulations": mcts["simulations"],
            "answer_preview": mcts["final_answer"][:200],
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 35: Advanced Reasoning SOLUTION ===\n")

    COMPLIANCE_PROBLEM = (
        "A US fintech company wants to expand to the EU. They process payment data for 2M users. "
        "They use a US-based cloud provider. They have SOC2 Type II but not ISO 27001. "
        "What are their top 3 compliance priorities, in what order should they tackle them, "
        "and what is the minimum timeline to become GDPR-compliant?"
    )

    print("1. Tree of Thought BFS (quick demo — depth=2, branching=2):")
    tot = await tree_of_thought_bfs(
        COMPLIANCE_PROBLEM, branching_factor=2, max_depth=2, beam_width=2
    )
    print(f"  Score: {tot['best_score']:.2f}")
    print(f"  Nodes explored: {tot['nodes_explored']}")
    print(f"  Answer: {tot['final_answer'][:300]}...\n")

    print("2. o3-mini extended thinking:")
    o3 = await o3_mini_reasoning(COMPLIANCE_PROBLEM, reasoning_effort="high")
    print(f"  Method: {o3['method']}")
    print(f"  Answer: {o3['answer'][:300]}...\n")

    print("3. MCTS reasoning (10 simulations):")
    mcts = await mcts_reasoning(COMPLIANCE_PROBLEM, simulations=10, branching_factor=2, max_depth=2)
    print(f"  Best avg score: {mcts['best_avg_score']:.2f}")
    print(f"  Answer: {mcts['final_answer'][:300]}...\n")

    print("4. Key trade-offs:")
    print("  Tree of Thought BFS: systematic, deterministic, memory-intensive O(b^d)")
    print("  o3-mini: delegated to model, highest quality, latency ~30s, expensive")
    print("  MCTS: efficient exploration via UCB1, stochastic, anytime algorithm")
    print("  → Use ToT for structured decisions, o3 for hard problems, MCTS for large trees")

if __name__ == "__main__":
    asyncio.run(main())
