"""
Exercise 3: Reflexion — Multi-Episode Learning from Failure
Goal: Unlike single-shot self-reflection, Reflexion persists lessons across episodes.

Pattern:
  Episode 1: Attempt task → Evaluate → Extract lesson → Store in memory
  Episode 2: Load memory → Attempt with lessons → Evaluate → Update memory
  Episode 3: ...

Tasks:
  1. Complete ReflexionMemory.add_episode() — store episode result + lesson.
  2. Complete ReflexionMemory.build_context() — format memory into a system prompt section.
  3. Complete derive_lesson() — ask LLM to extract a lesson from a failed attempt.
  4. Complete evaluate_answer() — score the answer 1-10, return (score, feedback).
  5. Complete run_reflexion() — run N episodes, passing accumulated lessons as context.
  6. Print score progression across episodes — it should improve!

Expected output:
  Episode 1: score=4 | lesson: "Forgot to include concrete examples"
  Episode 2: score=6 | lesson: "Examples better but missing edge cases"
  Episode 3: score=8 | lesson: "Good, but could be more concise"
  Score progression: 4 → 6 → 8 ✓
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


# ── Memory ─────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    episode_num: int
    task: str
    answer: str
    score: int
    feedback: str
    lesson: str


@dataclass
class ReflexionMemory:
    episodes: list[Episode] = field(default_factory=list)

    def add_episode(self, episode: Episode):
        """Store a completed episode."""
        # TODO: self.episodes.append(episode)
        raise NotImplementedError

    def build_context(self) -> str:
        """
        Format past episodes into a prompt section the agent can learn from.
        Return empty string if no episodes yet.
        TODO: For each episode, include: episode number, score, lesson learned.
        Format:
          [REFLEXION MEMORY]
          Episode 1 (score 4/10): Lesson: "..."
          Episode 2 (score 6/10): Lesson: "..."
          Apply these lessons in your current attempt.
        """
        raise NotImplementedError


# ── Core Functions ─────────────────────────────────────────────────────────────

EVALUATOR_SYSTEM = """You are a strict quality evaluator. Score the response 1-10 and give specific feedback.
Return ONLY JSON: {"score": <1-10>, "feedback": "<specific critique in 1-2 sentences>"}
Scoring guide: 1-3=poor, 4-5=below average, 6-7=acceptable, 8-9=good, 10=excellent"""

LESSON_SYSTEM = """You are a meta-learner. Given a task, an attempt, and evaluation feedback,
extract ONE concise lesson (max 20 words) that would improve the next attempt.
Return ONLY the lesson as a plain string. No preamble."""


def evaluate_answer(task: str, answer: str) -> tuple[int, str]:
    """Score the answer 1-10. Return (score, feedback)."""
    prompt = f"Task: {task}\n\nAnswer:\n{answer}"
    # TODO: call chat() with EVALUATOR_SYSTEM
    # TODO: parse JSON from get_text(response)
    # TODO: return (data["score"], data["feedback"])
    raise NotImplementedError


def derive_lesson(task: str, answer: str, feedback: str) -> str:
    """Extract a lesson from failure to use in the next attempt."""
    prompt = f"Task: {task}\n\nAttempt:\n{answer}\n\nFeedback: {feedback}"
    # TODO: call chat() with LESSON_SYSTEM
    # TODO: return get_text(response).strip()
    raise NotImplementedError


def attempt_task(task: str, memory: ReflexionMemory) -> str:
    """Attempt the task, incorporating past lessons from memory."""
    context = memory.build_context()
    system = "You are an expert writer and analyst. Produce high-quality, detailed responses."
    if context:
        system += f"\n\n{context}"
    response = chat([{"role": "user", "content": task}], system=system, max_tokens=600)
    return get_text(response)


# ── Reflexion Loop ─────────────────────────────────────────────────────────────

def run_reflexion(task: str, num_episodes: int = 3, target_score: int = 8) -> str:
    memory = ReflexionMemory()
    scores = []
    best_answer = ""
    best_score = 0

    print(f"\nTask: {task}")
    print(f"Running {num_episodes} Reflexion episodes (target score: {target_score}/10)\n")

    for ep in range(1, num_episodes + 1):
        print(f"Episode {ep}/{num_episodes}...")

        # Attempt
        answer = attempt_task(task, memory)

        # Evaluate
        score, feedback = evaluate_answer(task, answer)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_answer = answer

        print(f"  Score: {score}/10 | {feedback[:80]}")

        if score >= target_score:
            print(f"  ✅ Target reached!")
            break

        # Extract lesson (only if more episodes remain)
        if ep < num_episodes:
            lesson = derive_lesson(task, answer, feedback)
            print(f"  Lesson: {lesson}")

            episode = Episode(
                episode_num=ep,
                task=task,
                answer=answer,
                score=score,
                feedback=feedback,
                lesson=lesson,
            )
            memory.add_episode(episode)

    print(f"\nScore progression: {' → '.join(str(s) for s in scores)}")
    trend = "✓ Improving" if len(scores) > 1 and scores[-1] > scores[0] else "~ Flat"
    print(f"Trend: {trend} | Best score: {best_score}/10")
    return best_answer


TASKS = [
    "Explain gradient descent to a 10-year-old using a concrete analogy",
    "Write a 5-bullet executive summary of why companies adopt microservices",
    "Describe 3 real-world applications of reinforcement learning with outcomes",
]

if __name__ == "__main__":
    import sys as _sys
    task = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else TASKS[0]
    final = run_reflexion(task, num_episodes=3, target_score=8)
    print(f"\n{'='*50}\nBEST ANSWER:\n{final}")
