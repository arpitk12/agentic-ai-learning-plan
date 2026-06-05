"""
SOLUTION — Exercise 3: Reflexion — Multi-Episode Learning from Failure
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


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
        self.episodes.append(episode)

    def build_context(self) -> str:
        if not self.episodes:
            return ""
        lines = ["[REFLEXION MEMORY]"]
        for ep in self.episodes:
            lines.append(f'Episode {ep.episode_num} (score {ep.score}/10): Lesson: "{ep.lesson}"')
        lines.append("Apply these lessons in your current attempt.")
        return "\n".join(lines)


EVALUATOR_SYSTEM = """You are a strict quality evaluator. Score the response 1-10 and give specific feedback.
Return ONLY JSON: {"score": <1-10>, "feedback": "<specific critique in 1-2 sentences>"}
Scoring guide: 1-3=poor, 4-5=below average, 6-7=acceptable, 8-9=good, 10=excellent"""

LESSON_SYSTEM = """You are a meta-learner. Given a task, an attempt, and evaluation feedback,
extract ONE concise lesson (max 20 words) that would improve the next attempt.
Return ONLY the lesson as a plain string. No preamble."""


def evaluate_answer(task: str, answer: str) -> tuple[int, str]:
    prompt = f"Task: {task}\n\nAnswer:\n{answer}"
    response = chat(
        [{"role": "user", "content": prompt}],
        system=EVALUATOR_SYSTEM,
        max_tokens=200,
    )
    raw = get_text(response)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return int(data["score"]), data["feedback"]
    except Exception:
        return 5, "Could not parse evaluation"


def derive_lesson(task: str, answer: str, feedback: str) -> str:
    prompt = f"Task: {task}\n\nAttempt:\n{answer}\n\nFeedback: {feedback}"
    response = chat(
        [{"role": "user", "content": prompt}],
        system=LESSON_SYSTEM,
        max_tokens=60,
    )
    return get_text(response).strip()


def attempt_task(task: str, memory: ReflexionMemory) -> str:
    context = memory.build_context()
    system = "You are an expert writer and analyst. Produce high-quality, detailed responses."
    if context:
        system += f"\n\n{context}"
    response = chat([{"role": "user", "content": task}], system=system, max_tokens=600)
    return get_text(response)


def run_reflexion(task: str, num_episodes: int = 3, target_score: int = 8) -> str:
    memory = ReflexionMemory()
    scores = []
    best_answer = ""
    best_score = 0

    print(f"\nTask: {task}")
    print(f"Running {num_episodes} Reflexion episodes (target score: {target_score}/10)\n")

    for ep in range(1, num_episodes + 1):
        print(f"Episode {ep}/{num_episodes}...")
        answer = attempt_task(task, memory)
        score, feedback = evaluate_answer(task, answer)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_answer = answer

        print(f"  Score: {score}/10 | {feedback[:80]}")

        if score >= target_score:
            print("  ✅ Target reached!")
            break

        if ep < num_episodes:
            lesson = derive_lesson(task, answer, feedback)
            print(f"  Lesson: {lesson}")
            memory.add_episode(Episode(
                episode_num=ep, task=task, answer=answer,
                score=score, feedback=feedback, lesson=lesson,
            ))

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
