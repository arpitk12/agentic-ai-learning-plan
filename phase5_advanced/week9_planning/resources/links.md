# Week 9 Resources — Planning & Self-Correction

## Papers
- Reflexion (2023): https://arxiv.org/abs/2303.11366
- Tree of Thought (2023): https://arxiv.org/abs/2305.10601
- LATS — LLM as Tree Search (2023): https://arxiv.org/abs/2310.04406
- Plan-and-Solve Prompting (2023): https://arxiv.org/abs/2305.04091
- Self-Refine (2023): https://arxiv.org/abs/2303.17651

## Courses
- DeepLearning.AI: "AI Agentic Design Patterns with AutoGen": https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/

## Key Patterns
- **Self-Reflection**: generate → critique → regenerate (single attempt context)
- **Reflexion**: persist failure logs across episodes for long-term improvement  
- **Tree of Thought**: branch multiple reasoning paths, pick best
- **Plan-and-Execute**: full plan upfront, then execute each step

## When to Use Each
| Pattern | Use When |
|---|---|
| Self-reflection | Quality matters more than speed |
| Reflexion | Task repeats many times (optimization) |
| Tree of Thought | Complex multi-step reasoning |
| Plan-and-Execute | Steps are independent, parallelizable |
