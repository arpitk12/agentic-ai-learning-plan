# Project 15 — WhatsApp Multi-Agent Assistant (MCP + RAG)

## What You Build

A **production-grade conversational AI assistant** reachable from a phone via WhatsApp
(or Telegram). User messages are routed through a multi-agent system backed by a
**Model Context Protocol (MCP) server** that exposes RAG search, product lookup,
order status, and reminder creation as structured tools. A session manager maintains
per-user conversation history across messages.

---

## Architecture

```
📱 WhatsApp / Telegram (your phone)
          │
          ▼ POST /webhook/twilio  or  POST /webhook/telegram
┌─────────────────────────────────────────────────────────┐
│                   FastAPI App (port 8000)                │
│                                                         │
│   SessionManager (phone → conversation history)         │
│                │                                        │
│   classify_intent()   ←  LLM triage (5 categories)     │
│         │                                               │
│    ┌────┴────────────────────────┐                      │
│    ▼         ▼         ▼        ▼                       │
│  RAGAgent  ProductAgent OrderAgent GeneralAgent         │
│    │           │           │                            │
│    └─────────┬─┘           │                            │
│              ▼             ▼                            │
│     ┌────────────────────────────┐                      │
│     │  MCP Client (stdio)        │                      │
│     └────────────┬───────────────┘                      │
└──────────────────│──────────────────────────────────────┘
                   │ stdio (subprocess)
         ┌─────────▼──────────────┐
         │   MCP Server           │
         │   (mcp_server.py)      │
         │                        │
         │ ● search_knowledge_base │ ← TF-IDF RAG over 15 KB docs
         │ ● get_product_info      │ ← product catalog
         │ ● get_order_status      │ ← mock order DB
         │ ● create_reminder       │ ← in-memory reminder store
         └────────────────────────┘
                   │
          response → LLM synthesises → reply → Twilio/Telegram API → 📱
```

---

## Production Patterns Covered

| Pattern | Where |
|---------|-------|
| **MCP server** (FastMCP, tool registration, schema) | `mcp_server.py` |
| **MCP client** (stdio transport, `ClientSession`, lifespan management) | `starter.py` — `lifespan()` |
| **RAG via MCP tool** (TF-IDF retrieval, top-k chunks → LLM) | `search_knowledge_base` tool + `rag_agent()` |
| **Multi-agent routing** (intent → specialist) | `classify_intent()` + `route_message()` |
| **Per-user session state** (phone/chat ID → history) | `SessionManager` |
| **WhatsApp webhook** (Twilio TwiML response) | `POST /webhook/twilio` |
| **Telegram webhook** (Bot API, `sendMessage`) | `POST /webhook/telegram` |
| **Platform-agnostic agent core** (same agents, different I/O) | `route_message(platform=...)` |
| **CLI demo mode** (no credentials needed) | `cli_demo()` |
| Guide reference | §2 (tools/MCP), §3 (RAG), §4 (multi-agent), §7 (production API) |

---

## Knowledge Base (TechFlow SaaS — 15 embedded docs)

| Topic | KB IDs |
|-------|--------|
| Product plans (Basic / Pro / Enterprise) | kb001–kb003 |
| Pricing and trials | kb004 |
| Password reset, 2FA, security | kb005–kb006 |
| API rate limits and integrations | kb007–kb008 |
| Data export, billing, cancellation | kb009–kb011 |
| Storage limits, team management | kb012–kb013 |
| Mobile app, support SLA | kb014–kb015 |

---

## Milestones

### Milestone 1 — MCP Server: RAG Tool
In `mcp_server.py`, implement `_tokenise()` (word tokeniser) and
`search_knowledge_base()` (TF-IDF cosine scoring over `_TF_LIST` / `_IDF`,
return top-3 formatted results). Verify you can run `python mcp_server.py`
and it starts without errors.

### Milestone 2 — MCP Server: Lookup Tools
Implement `get_product_info()` (dict lookup in `PRODUCTS`, format nicely),
`get_order_status()` (lookup in `ORDERS`, include status + next billing date),
and `create_reminder()` (append to `REMINDERS`, return confirmation string).

### Milestone 3 — MCP Client Lifespan
In `starter.py`, implement `lifespan(app)`:
- Create `StdioServerParameters` pointing at `mcp_server.py`
- Open `stdio_client` + `ClientSession` using `async with`
- `await session.initialize()`
- Store session in `app.state.mcp` and `app.state.sessions`
- Yield to let the app run; cleanup happens automatically

### Milestone 4 — Intent Classifier + Specialist Agents
Implement `classify_intent()` (LLM → one of `rag|product|order|reminder|general`).
Then implement all 4 specialist agents:
- `rag_agent()` — call `search_knowledge_base` via MCP, pass chunks to LLM
- `product_agent()` — call `get_product_info` via MCP
- `order_agent()` — extract order ID from message, call `get_order_status`
- `general_agent()` — direct LLM with conversation history

### Milestone 5 — Webhooks and Routing
Implement `route_message()` (orchestrator: classify → dispatch → return reply).
Implement `parse_twilio()` (extract `From` + `Body` from Twilio form data) and
`parse_telegram()` (extract `chat.id` + `text` from Telegram JSON).
Wire both POST endpoints to use `route_message()`.

### Milestone 6 — Messaging Platform Integration
Set up one or both platforms:

**WhatsApp via Twilio:**
1. Create free Twilio account → https://console.twilio.com
2. Messaging → Try it out → Send a WhatsApp message → join sandbox
3. `ngrok http 8000` → copy the HTTPS URL
4. Set webhook: `https://<ngrok-url>/webhook/twilio`
5. Set env vars: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_NUMBER`

**Telegram:**
1. Open Telegram → message `@BotFather` → `/newbot` → copy the token
2. `ngrok http 8000` → copy HTTPS URL
3. Register webhook: `curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=<ngrok-url>/webhook/telegram"`
4. Set env var: `TELEGRAM_BOT_TOKEN`

---

## Expected Output (CLI Demo Mode)

```
═══════════════════════════════════════════════════════════════════
 Project 15 — WhatsApp Multi-Agent (MCP + RAG)   [gemini/gemini-2.0-flash]
═══════════════════════════════════════════════════════════════════
MCP server started (PID 12345)  ✅
MCP tools available: search_knowledge_base, get_product_info, get_order_status, create_reminder
FastAPI running on http://0.0.0.0:8000

Running in CLI demo mode (set TWILIO or TELEGRAM credentials for messaging)
Type 'quit' to exit, 'clear' to reset session

You: What's included in the Pro plan?
[intent: product]  [tool: get_product_info]
Agent: TechFlow Pro includes unlimited projects, 100GB storage, priority support,
       and advanced analytics — $99/month or $990/year (save 2 months). 14-day free trial!

You: What are the API rate limits?
[intent: rag]  [tool: search_knowledge_base → kb007]
Agent: On the Pro plan you get 1,000 requests/minute and 100,000/day. Rate limit headers
       (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset) are included in all API responses.

You: Remind me to cancel before my trial ends
[intent: reminder]  [tool: create_reminder]
Agent: Reminder set! ✅ I'll note: "Cancel before trial ends" for you (ID: R-001).
```

---

## Setup

```bash
# Install dependencies
pip install litellm python-dotenv pydantic fastapi uvicorn httpx mcp

# Optional — only needed for real WhatsApp integration
pip install twilio

# Run (CLI demo — no credentials needed)
python starter.py                   # or: python solution/solution.py

# Run as API server (set credentials in .env first)
uvicorn starter:app --host 0.0.0.0 --port 8000 --reload

# Test the /chat endpoint directly
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What plans do you offer?", "user_id": "test123"}'
```

> ⚠️ **The MCP server (`mcp_server.py`) is started automatically** as a subprocess
> when the FastAPI app starts — you do not need to run it separately.
