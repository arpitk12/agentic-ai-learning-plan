# Week 1 — LLM API Mastery

## What This Week Is About
Before building agents, you need to speak the language of LLMs fluently. This week covers the core mechanics: how LLMs receive input, how they produce output, and how to control both precisely. Every agent you build later is ultimately a loop around these primitives.

---

## 1. The Message Format — How LLMs Think

Every LLM call is a **conversation** expressed as an ordered list of role-tagged messages. The model has no persistent state — it sees the full history on every call.

```python
messages = [
    {"role": "system",    "content": "You are a concise assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "And Germany?"},
]
```

### Roles Explained

| Role | Purpose | Key Point |
|------|---------|-----------|
| `system` | Sets model persona, rules, output format | Most powerful lever. Read before every user message. |
| `user` | Human's input | Can include code, data, files |
| `assistant` | Model's previous response | Pre-filling this steers output format |
| `tool` | Result of a function call | Injected by you after executing a tool |

### Why This Matters for Agents
Agents are programs that keep appending messages to this list, calling the LLM in a loop. The message list IS the agent's working memory. Understanding the format is understanding how agents work at their core.

---

## 2. LiteLLM — The Universal LLM Interface

**LiteLLM** is a Python library providing a single API to 100+ LLM providers: OpenAI, Anthropic, Google, Groq, Cohere, Azure, Mistral, and more.

**Purpose**: Write your agent code once. Switch providers by changing one environment variable (`MODEL=gemini/gemini-2.0-flash` → `MODEL=openai/gpt-4o`). No code changes needed.

```python
import litellm

response = litellm.completion(
    model="gemini/gemini-2.0-flash",   # ← change this to switch providers
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Supported Model Strings
```
"gemini/gemini-2.0-flash"        # Google — fast, cheap, large context
"openai/gpt-4o"                  # OpenAI — reliable, widely supported
"openai/gpt-4o-mini"             # OpenAI — cheap, fast
"anthropic/claude-3-5-sonnet"    # Anthropic — excellent reasoning
"groq/llama-3.3-70b-versatile"   # Groq — extremely fast inference
"ollama/llama3.2"                # Local — free, private
```

### Our `llm.py` Wrapper
The repo root has `llm.py` — a thin wrapper over LiteLLM that normalizes responses:

```python
from llm import chat, get_text, stream_chat, get_tool_calls, stop_reason, calc_cost, MODEL

response = chat(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    system="Be concise. Answer in one word.",
    max_tokens=10,
)
text = get_text(response)          # "4"
cost = calc_cost(MODEL, 20, 5)     # e.g. 0.0000012
```

**Why a wrapper?** Different providers shape their response objects differently. Our wrapper normalizes them so `get_text()`, `get_tool_calls()`, and `stop_reason()` work identically across all providers.

---

## 3. Prompt Engineering — Getting the Output You Want

### System Prompts
The system prompt is the most powerful control lever. It sets persona, constraints, output format, and behavioral rules:

```
You are a senior Python engineer at a fintech company reviewing pull requests.
Rules:
- Flag any use of eval() or exec() as CRITICAL severity
- Point out missing type annotations
- Suggest one improvement per function maximum
- Always respond in JSON: {"severity": "critical|high|medium|low", "issues": [...], "suggestions": [...]}
Do NOT include markdown formatting. JSON only.
```

**Tips for strong system prompts:**
- Be explicit, not vague. "Be helpful" → weak. "Answer in ≤3 sentences, use bullet points for lists" → strong.
- Put the most important rules at the **start AND end** (LLMs attend more to these positions).
- Use XML-style delimiters to separate sections: `<context>`, `<rules>`, `<output_format>`.
- Include examples of what good output looks like ("Few-shot in the system prompt").

### Chain-of-Thought (CoT)
Force the model to reason step-by-step before answering:

```python
system = """Think through the problem step by step inside <thinking> tags.
Then provide your final answer inside <answer> tags.
Example:
<thinking>First I need to... then I should... therefore...</thinking>
<answer>42</answer>"""
```

CoT dramatically improves accuracy on math, logic, and multi-step reasoning. The cost: more output tokens.

### Zero-Shot CoT
The simplest form — just append `"Let's think step by step."` to your prompt. Shown to improve reasoning accuracy by 40-60% on benchmarks. Free performance gain.

### Few-Shot Prompting
Show examples of the desired input/output pattern before asking:

```
Input: "I love this product! Works perfectly."
Output: {"sentiment": "positive", "score": 0.95}

Input: "Completely broken, terrible support."
Output: {"sentiment": "negative", "score": 0.05}

Input: "It's okay, nothing special."
Output:
```

The model completes the pattern. Especially effective for unusual or precise output formats.

---

## 4. Context Windows & Token Management

### What a Token Is
A token ≈ 0.75 words in English. "The quick brown fox" ≈ 5 tokens. Code is more token-dense (identifiers, punctuation all count). Use [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) to count precisely.

### Context Window Limits (2025)

| Model | Context | Cost (input/1K tokens) |
|-------|---------|----------------------|
| Gemini 2.0 Flash | 1M tokens | ~$0.0001 |
| GPT-4o | 128K tokens | ~$0.005 |
| Claude 3.5 Sonnet | 200K tokens | ~$0.003 |
| Llama 3.3 70B (Groq) | 128K tokens | ~$0.0001 |

### The Critical Rule
**The model sees ALL messages on EVERY call.** A 20-turn conversation × 300 tokens/turn = 6,000 tokens in overhead before your question is even asked. Costs compound fast in long agent loops.

### Context Management Strategies

**1. Sliding Window** — keep only the last N messages:
```python
MAX_HISTORY = 20
if len(messages) > MAX_HISTORY:
    messages = messages[:1] + messages[-(MAX_HISTORY - 1):]  # keep system + last N
```

**2. Rolling Summary** — compress old turns every K messages:
```python
if len(messages) > 30:
    old = messages[1:20]  # skip system, take old messages
    summary = summarize(old)
    messages = messages[:1] + [{"role": "system", "content": f"Prior conversation summary: {summary}"}] + messages[20:]
```

**3. Retrieval** — store history in vector DB, retrieve only relevant turns (see Week 4).

---

## 5. Structured Output — Forcing JSON

### Method 1: Prompt Engineering
```
Respond ONLY with valid JSON. No markdown, no explanation.
Schema: {"name": string, "age": integer, "skills": [string]}
```

### Method 2: Pydantic + Retry
```python
from pydantic import BaseModel, ValidationError
import json, re

class PersonProfile(BaseModel):
    name: str
    age: int
    skills: list[str]

def get_profile(description: str) -> PersonProfile:
    prompt = f"Extract a PersonProfile from: {description}\nReturn JSON only."
    for attempt in range(3):
        raw = get_text(chat([{"role": "user", "content": prompt}]))
        # Strip markdown fences if present
        clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
        try:
            return PersonProfile(**json.loads(clean))
        except (json.JSONDecodeError, ValidationError) as e:
            prompt += f"\n\nYour previous response failed validation: {e}. Fix it."
    raise ValueError("Could not parse after 3 attempts")
```

### Method 3: JSON Mode (OpenAI/some providers)
```python
response = litellm.completion(
    model="openai/gpt-4o",
    messages=[...],
    response_format={"type": "json_object"},  # guarantees valid JSON
)
```

---

## 6. Streaming — Real-Time Token Display

Streaming returns tokens as generated instead of waiting for the complete response. Essential for good UX.

```python
from llm import stream_chat

print("Assistant: ", end="")
for chunk in stream_chat([{"role": "user", "content": "Tell me a story"}]):
    print(chunk, end="", flush=True)
print()
```

**When to use streaming:**
- ✅ User-facing chat interfaces — feels faster, more responsive
- ✅ Long responses where the user reads as it generates
- ❌ When you need the full response before proceeding (tool calls, JSON parsing)

---

## 7. Cost Estimation

```python
from llm import calc_cost, MODEL

# After any response:
in_tok = response.usage.prompt_tokens
out_tok = response.usage.completion_tokens
cost = calc_cost(MODEL, in_tok, out_tok)
print(f"This call: ${cost:.6f}")

# Quick mental model:
# Gemini 2.0 Flash:  1M tokens ≈ $0.10–$0.40  → extremely cheap
# GPT-4o-mini:       1M tokens ≈ $0.15–$0.60  → cheap
# Claude Haiku:      1M tokens ≈ $0.80–$4.00  → moderate
# GPT-4o / Sonnet:   1M tokens ≈ $5–$15       → expensive
```

**Rule**: Always track cost per agent run. A 10-step agent loop on GPT-4o can cost $0.05–$0.20. At scale (1000 users/day) that's $50–$200/day on one endpoint alone.

---

## Tools & Libraries Used This Week — Deep Dive

Understanding each tool's purpose prevents cargo-culting. Here's exactly what each tool is, why you need it, and what breaks without it.

### LiteLLM — Why It's the Foundation of Everything

**The problem it solves**: Every LLM provider has a different API. OpenAI uses `client.chat.completions.create()`. Anthropic uses `client.messages.create()`. Google uses `genai.generate_content()`. Each has different parameter names, response formats, and error types.

**Without LiteLLM**: Your code is locked to one provider. Switching from GPT-4o to Gemini requires rewriting every LLM call in your codebase. This is vendor lock-in at its worst.

**With LiteLLM**: One API, 100+ providers. Change `MODEL=openai/gpt-4o` to `MODEL=gemini/gemini-2.0-flash` in your `.env` and every single call works identically.

```python
# What LiteLLM does under the hood:
# 1. Reads your model string ("gemini/gemini-2.0-flash")
# 2. Routes to the correct provider SDK (google-generativeai)
# 3. Translates your messages format to provider format
# 4. Makes the API call
# 5. Translates the response BACK to the OpenAI format
# 6. Returns it to you — always the same shape regardless of provider

import litellm

# All of these produce identical response objects:
r1 = litellm.completion("openai/gpt-4o", messages=[...])
r2 = litellm.completion("gemini/gemini-2.0-flash", messages=[...])
r3 = litellm.completion("anthropic/claude-3-5-sonnet", messages=[...])

# Access text the same way regardless of provider
text = r1.choices[0].message.content  # works for all three
```

**When you'd NOT use LiteLLM**: Almost never. The rare exception: you need a provider-specific feature that LiteLLM hasn't implemented yet (e.g., very new Anthropic features). Even then, LiteLLM's passthrough mechanism usually handles it.

---

### Pydantic — Why Agents Need Data Validation

**The problem it solves**: LLMs return text. But you need structured data. Without validation, a missing field or wrong type crashes your agent at runtime, often 10 steps into a long task.

**What Pydantic does**:
1. Defines the shape of data with Python type annotations
2. Validates at runtime that data matches the schema
3. Gives clear error messages when validation fails (not `KeyError: 'name'` but `name field required`)
4. Converts types automatically (string "42" → integer 42)

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)  # must be 0-1
    key_phrases: list[str]
    
    # Custom validation
    @field_validator("key_phrases")
    @classmethod
    def max_five_phrases(cls, v):
        return v[:5]  # silently limit to 5

# Pydantic catches errors BEFORE they propagate
try:
    result = SentimentResult(sentiment="happy", confidence=1.5, key_phrases=["great"])
except ValidationError as e:
    print(e)
    # sentiment: Input should be 'positive', 'negative' or 'neutral' [type=literal_error]
    # confidence: Input should be less than or equal to 1 [type=less_than_equal]
```

**FastAPI + Pydantic**: FastAPI uses Pydantic to validate ALL incoming HTTP requests and outgoing responses. If you use `BaseModel` for your request body, FastAPI automatically returns a 422 error with clear field-by-field messages when validation fails.

---

### python-dotenv — Managing Secrets Safely

**The problem it solves**: API keys must never be in your code or git repository. `python-dotenv` lets you store them in a `.env` file (which is gitignored) and loads them as environment variables.

```bash
# .env file (NEVER commit this to git)
GEMINI_API_KEY=AIzaSy...your_key_here
MODEL=gemini/gemini-2.0-flash
API_KEY=my-secret-api-key
REDIS_URL=redis://localhost:6379/0
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env into environment variables

api_key = os.getenv("GEMINI_API_KEY")  # now available
model = os.getenv("MODEL", "gemini/gemini-2.0-flash")  # with default
```

**Why environment variables?**: In production (Docker, Kubernetes, Cloud), secrets are injected as environment variables by the orchestrator. `python-dotenv` makes local dev work the same way — your code reads from env vars in both cases.

---

### `llm.py` — Our Repo's Wrapper

**Why we don't call LiteLLM directly everywhere**: LiteLLM's response format still requires boilerplate (`.choices[0].message.content`). Our wrapper normalizes this and adds:
- `get_text(response)`: extracts text content
- `get_tool_calls(response)`: extracts tool call list as `[{id, name, arguments}]`
- `stop_reason(response)`: normalizes to `"tool_calls"` or `"stop"` across providers
- `calc_cost(model, prompt_tokens, completion_tokens)`: cost in USD
- `assistant_message(response)`: properly formats for message history
- `tool_result_message(call_id, result)`: properly formats tool result

```python
# Without wrapper (verbose, error-prone)
response = litellm.completion(model=MODEL, messages=messages)
content = response.choices[0].message.content
if response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        # ... handle each

# With llm.py wrapper (clean, consistent)
response = chat(messages)
if stop_reason(response) == "tool_calls":
    for tc in get_tool_calls(response):
        name = tc["name"]     # already a string
        args = tc["arguments"]  # already a dict
```

---

## How Tokens Work — The Technical Reality

Tokens are the atomic unit of LLM input/output. Understanding tokens helps you:
- Predict costs accurately
- Debug context window errors
- Optimize prompt length

**Tokenization rules** (GPT-family, approximate):
- Common English words: 1 token each (`the`, `is`, `Python`)
- Less common words: 2-3 tokens (`tokenization` = `token` + `ization`)
- Code: more tokens per character (punctuation, underscores each often 1 token)
- Chinese/Japanese/Korean: typically 2-3 tokens per character
- Empty space: often its own token

```python
# Exact token counting with tiktoken (OpenAI's tokenizer)
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

texts = [
    "Hello world",                     # 2 tokens
    "The quick brown fox",             # 4 tokens
    "def calculate_fibonacci(n):",     # 8 tokens (punctuation + underscores)
    "你好世界",                          # 6 tokens (Chinese)
]

for text in texts:
    n = len(encoder.encode(text))
    print(f"{n:3d} tokens: {text!r}")
```

**Why this matters for cost**:
- Gemini 2.0 Flash: ~$0.10 per 1M input tokens, ~$0.40 per 1M output tokens
- If your system prompt is 500 tokens and you make 1000 requests/day:
  - 500 × 1000 = 500K tokens/day just from the system prompt
  - At Gemini's price: ~$0.05/day — almost negligible
  - Same on GPT-4o ($5/1M): $2.50/day from system prompt alone

---

## Prompting — The Science and Art

### Why System Prompts Are So Powerful

The system prompt is processed before every user message. It's the model's "mental state" going into the conversation. Good system prompts:

1. **Set expectations**: "You are a Python expert" → model allocates attention to Python-related concepts
2. **Constrain output**: "Always respond in JSON" → enforces format without hoping the model remembers
3. **Control tone**: "Be concise, max 3 sentences" → enforces brevity
4. **Define tools**: In tool-using agents, the system prompt explains tool purpose

**The recency effect**: Models pay more attention to content at the beginning and end of prompts. Put the most critical rules at the START (so they're always read first) and sometimes repeat key constraints at the END.

### Temperature and Sampling Parameters

```python
# Temperature controls randomness
# 0.0 = deterministic (always picks highest-probability token)
# 1.0 = balanced creativity/coherence
# 2.0 = very random/creative

chat(messages, temperature=0.0)   # deterministic — use for: JSON extraction, code, classification
chat(messages, temperature=0.3)   # slight variation — use for: summaries, analysis
chat(messages, temperature=0.7)   # creative — use for: writing, brainstorming (default)
chat(messages, temperature=1.2)   # very creative — use for: creative fiction

# Top-p (nucleus sampling) — alternative to temperature
chat(messages, top_p=0.9)         # only considers tokens that sum to 90% probability mass

# Max tokens — hard limit on response length
chat(messages, max_tokens=100)    # truncates response at 100 tokens
chat(messages, max_tokens=4096)   # allows long responses (but costs more)

# Stop sequences — stop generation at specific strings
chat(messages, stop=["</answer>", "###"])  # stop when these strings appear
```

### Chain-of-Thought: The Mechanism

CoT works because transformers process tokens sequentially. When the model writes out reasoning steps, those tokens become part of its context when generating the answer. More computation happens (more tokens generated), which tends to produce more accurate answers.

```python
# Research finding: "Let's think step by step" increases accuracy by ~40-60%
# on GSM8K math benchmark (Kojima et al., 2022)

# Zero-shot CoT (simplest)
messages = [{"role": "user", "content": f"{your_question}\n\nLet's think step by step."}]

# Structured CoT (more controlled)
system = """Think through the problem in <thinking> tags, then give your final answer in <answer> tags.

<thinking>
[Work through the problem here — show all steps]
</thinking>
<answer>
[Just the final answer here]
</answer>"""

# Self-consistency CoT (most accurate, highest cost)
# Generate 5 independent CoT answers, take the majority
def self_consistent_answer(question: str, n: int = 5) -> str:
    answers = []
    for _ in range(n):
        resp = get_text(chat([{"role": "user", "content": f"{question}\n\nLet's think step by step."}],
                             temperature=0.7))
        # Extract final answer from response
        answers.append(resp.split("\n")[-1])
    
    # Majority vote
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]
```

---

## Common Pitfalls — Week 1

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not loading `.env` | `GEMINI_API_KEY` is None, API call fails | Call `load_dotenv()` before any LLM call |
| Using wrong model string | `litellm.BadRequestError` | Check provider prefix: `gemini/`, `openai/`, etc. |
| Not stripping markdown from JSON | `json.JSONDecodeError` | `re.sub(r'```json?\s*|\s*```', '', raw)` |
| Infinite prompt growth | Context window exceeded (400 error) | Implement sliding window from the start |
| Temperature=0 for creative tasks | All responses identical | Use 0.7 for creative, 0.0 for structured |
| Not handling API errors | Agent crashes mid-run | Wrap all `chat()` calls in try/except with retry |
| Long system prompt with no structure | Model ignores later rules | Use `<rules>`, `<format>` XML tags for structure |
| Counting tokens by word | Off by 30-50% | Use `tiktoken` for accurate counts |
- `ex2_structured_output.py` — Pydantic parsing with retry
- `ex3_streaming.py` — live token streaming
- `ex4_prompt_comparison.py` — benchmark zero-shot vs few-shot vs CoT on a reasoning task

## Checklist
- [ ] Built multi-turn CLI chatbot with sliding window (last 20 messages)
- [ ] Forced JSON output, validated with Pydantic, retried on parse failure
- [ ] Implemented streaming — tokens print in real-time
- [ ] Compared 3 prompt strategies on the same math problem — recorded which scored best
- [ ] Tracked cost of every call; estimated cost for 1000 calls/day
