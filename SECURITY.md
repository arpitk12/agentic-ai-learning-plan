# Keeping API Keys Safe

This project handles real API keys. Follow these rules throughout all exercises.

---

## The Golden Rules

1. **Never hardcode a key in any `.py` file** — always use `os.getenv()`
2. **Never commit `.env`** — it is in `.gitignore`; only `.env.example` (no real keys) is committed
3. **Never print or log an API key** — not even "for debugging"
4. **Never paste a key into an LLM chat** (GitHub Copilot, ChatGPT, etc.) — they may log it
5. **Never share your screen** while `.env` is open in an editor

---

## Project Setup (Do This Once)

```bash
# 1. Copy the example — this is the only file with real keys
cp .env.example .env

# 2. Fill in your keys in .env (never in .env.example)
nano .env

# 3. Verify .env is gitignored
git check-ignore -v .env      # should print: .gitignore:2:.env
```

---

## Correct Pattern in Every Exercise

```python
# ✅ CORRECT — load from environment
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")   # None if not set, never hardcoded

# ❌ WRONG — never do this
api_key = "gsk_abc123xyz..."
```

`llm.py` already handles this for you — just use `from llm import chat`.

---

## Rotation (If a Key Is Exposed)

If you accidentally push a key to git or paste it somewhere:

1. **Immediately revoke it** in the provider dashboard:
   - Groq: [console.groq.com/keys](https://console.groq.com/keys)
   - Gemini: [aistudio.google.com](https://aistudio.google.com) → API Keys
   - Cerebras: [cloud.cerebras.ai](https://cloud.cerebras.ai) → API Keys
   - OpenRouter: [openrouter.ai/keys](https://openrouter.ai/keys)
2. Generate a new key and update your `.env`
3. If committed to git, scrub history with `git filter-repo` or BFG

---

## What `llm.py` Does to Protect You

- `litellm.suppress_debug_info = True` — prevents LiteLLM from printing keys in tracebacks
- `litellm.set_verbose = False` — disables HTTP request logging (which can contain auth headers)
- `_check_secrets_not_exposed()` — raises an error if a secret-looking string is passed as a CLI argument

---

## What NOT to Send to an LLM Assistant

When using GitHub Copilot, ChatGPT, Claude, etc. to help with exercises:

| ❌ Never share | ✅ Safe to share |
|---|---|
| Contents of `.env` | Contents of `.env.example` |
| Full tracebacks with auth headers | Error message without the header values |
| `os.environ` dumps | Specific env var *names* (not values) |
| Any string starting with `sk-`, `gsk_`, `AIza`, `csk_` | Model names, config flags |

---

## Environment Isolation Per Phase

As you progress through phases, you may accumulate many keys.
Consider using separate `.env` files per phase:

```bash
# Run with a specific env file
set -a && source .env.phase1 && set +a && python ex1.py

# Or use direnv (auto-loads .envrc per directory)
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
```

---

## Checklist Before Pushing to GitHub

```bash
# Check nothing sensitive is staged
git diff --cached | grep -iE "(api_key|secret|token|password)\s*=\s*\S+"

# Confirm .env is ignored
git status | grep .env    # should show nothing

# Scan for accidentally hardcoded keys
grep -rn "sk-ant\|gsk_\|AIza\|csk_\|sk-or-" --include="*.py" .
```
