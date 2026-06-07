"""
Project 15 — WhatsApp Multi-Agent Assistant: MCP + RAG (Solution)

Full implementation: FastAPI lifespan starts MCP server subprocess →
MCP client session → intent classification → specialist agents via MCP tools →
Twilio TwiML / Telegram Bot API responses → per-user session history.

Run modes:
  python solution.py                                    ← CLI demo (no credentials needed)
  uvicorn solution:app --host 0.0.0.0 --port 8000       ← full API server

Env vars (all optional — app works without them in CLI mode):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER
  TELEGRAM_BOT_TOKEN
"""

import os, sys, re, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from llm import achat, get_text, MODEL

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

# ══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    MAX_TURNS = 8

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}

    def get_history(self, user_id: str) -> list[dict]:
        return self._sessions.get(user_id, [])

    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append({"role": role, "content": content})
        cap = self.MAX_TURNS * 2
        if len(self._sessions[user_id]) > cap:
            self._sessions[user_id] = self._sessions[user_id][-cap:]

    def clear(self, user_id: str):
        self._sessions.pop(user_id, None)

    def history_as_text(self, user_id: str) -> str:
        msgs = self.get_history(user_id)[-6:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)

# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are TechFlow's helpful AI assistant, reachable via WhatsApp and Telegram. "
    "Be concise (phone-friendly: under 200 words), warm, and accurate. "
    "Use plain text — no markdown since messaging apps render it poorly."
)


async def _llm(messages: list[dict], system: str = _SYSTEM) -> str:
    r = await achat(messages, system=system, max_tokens=350)
    return get_text(r)

# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool Helper
# ══════════════════════════════════════════════════════════════════════════════

async def call_mcp_tool(mcp_session: ClientSession, tool: str, args: dict) -> str:
    result = await mcp_session.call_tool(tool, args)
    if result.content:
        return result.content[0].text
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

INTENT_CATEGORIES = ["rag", "product", "order", "reminder", "general"]

async def classify_intent(message: str, history_text: str) -> str:
    prompt = (
        "Classify this message into exactly one category.\n\n"
        "Categories:\n"
        "  rag      — knowledge base question: features, pricing, policies, how-to, security\n"
        "  product  — specific product comparison or plan recommendation\n"
        "  order    — order or subscription status (contains ORD-NNN or 'my order'/'my subscription')\n"
        "  reminder — user wants to set a reminder or note\n"
        "  general  — greetings, off-topic, or anything else\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"New message: {message}\n\n"
        "Reply with ONLY one word."
    )
    raw = (await _llm(
        [{"role": "user", "content": prompt}],
        system="You are a message classifier. Reply with exactly one word."
    )).strip().lower()
    return raw if raw in INTENT_CATEGORIES else "general"

# ══════════════════════════════════════════════════════════════════════════════
# Specialist Agents
# ══════════════════════════════════════════════════════════════════════════════

async def rag_agent(message: str, history: list[dict],
                    mcp_session: ClientSession) -> str:
    chunks = await call_mcp_tool(mcp_session, "search_knowledge_base", {"query": message})
    ctx    = chunks if chunks else "No specific knowledge base results found."
    messages = (history[-4:] if len(history) > 4 else history) + [
        {"role": "user", "content":
         f"Knowledge base results:\n{ctx}\n\nUser question: {message}\n\n"
         "Answer using the knowledge base results. Be concise and phone-friendly."}
    ]
    return await _llm(messages)


async def product_agent(message: str, history: list[dict],
                        mcp_session: ClientSession) -> str:
    # Determine which product the user is asking about
    msg_lower = message.lower()
    if "enterprise" in msg_lower:
        product_name = "techflow enterprise"
    elif "basic" in msg_lower:
        product_name = "techflow basic"
    else:
        product_name = "techflow pro"   # default to most common

    info = await call_mcp_tool(mcp_session, "get_product_info", {"product_name": product_name})
    messages = (history[-4:] if len(history) > 4 else history) + [
        {"role": "user", "content":
         f"Product info:\n{info}\n\nUser message: {message}\n\n"
         "Answer the user's product question based on the info above. Be concise."}
    ]
    return await _llm(messages)


async def order_agent(message: str, history: list[dict],
                      mcp_session: ClientSession) -> str:
    order_ids = re.findall(r'ORD-\d+', message, re.IGNORECASE)
    if order_ids:
        status = await call_mcp_tool(mcp_session, "get_order_status",
                                     {"order_id": order_ids[0].upper()})
        messages = (history[-4:] if len(history) > 4 else history) + [
            {"role": "user", "content":
             f"Order status info:\n{status}\n\nUser message: {message}\n\n"
             "Give a helpful, clear response about their order."}
        ]
        return await _llm(messages)
    else:
        return (
            "I'd be happy to check your order status! "
            "Please share your order ID (format: ORD-NNN) and I'll look it up right away."
        )


async def reminder_agent(message: str, user_id: str,
                         mcp_session: ClientSession) -> str:
    # Strip common reminder trigger phrases to extract the actual reminder text
    text = message.strip()
    for phrase in ["remind me to", "set a reminder to", "set a reminder for",
                   "remind me about", "remember to", "don't let me forget to"]:
        text = re.sub(rf'(?i){re.escape(phrase)}\s*', '', text).strip()
    text = text.rstrip(".!?").strip() or message

    result = await call_mcp_tool(mcp_session, "create_reminder",
                                 {"user_id": user_id, "text": text})
    return result   # the MCP tool already returns a confirmation string


async def general_agent(message: str, history: list[dict]) -> str:
    messages = (history[-6:] if len(history) > 6 else history) + [
        {"role": "user", "content": message}
    ]
    return await _llm(messages)

# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

_HELP_TEXT = (
    "TechFlow Assistant — Commands:\n"
    "  Ask any question about our product, pricing, or account.\n"
    "  'my order ORD-NNN' — check subscription status\n"
    "  'remind me to ...' — set a reminder\n"
    "  'clear' — reset conversation history"
)

async def route_message(message: str, user_id: str, platform: str,
                        sessions: SessionManager,
                        mcp_session: ClientSession) -> str:
    # Special commands (no LLM needed)
    if message.strip().lower() == "clear":
        sessions.clear(user_id)
        return "Conversation cleared. Fresh start!"
    if message.strip().lower() in ("help", "?"):
        return _HELP_TEXT

    sessions.add_message(user_id, "user", message)
    history = sessions.get_history(user_id)
    intent  = await classify_intent(message, sessions.history_as_text(user_id))

    print(f"  [{user_id[:12]:<12}] intent={intent:<9} platform={platform}")

    if intent == "rag":
        response = await rag_agent(message, history, mcp_session)
    elif intent == "product":
        response = await product_agent(message, history, mcp_session)
    elif intent == "order":
        response = await order_agent(message, history, mcp_session)
    elif intent == "reminder":
        response = await reminder_agent(message, user_id, mcp_session)
    else:
        response = await general_agent(message, history)

    sessions.add_message(user_id, "assistant", response)
    return response

# ══════════════════════════════════════════════════════════════════════════════
# Platform Parsers
# ══════════════════════════════════════════════════════════════════════════════

def parse_twilio(form_data: dict) -> tuple[str, str]:
    sender  = form_data.get("From", "").strip()
    message = form_data.get("Body", "").strip()
    if not sender or not message:
        return "", ""
    return sender, message


def parse_telegram(body: dict) -> tuple[str, str]:
    msg     = body.get("message", {})
    chat_id = str(msg.get("chat", {}).get("id", "")).strip()
    text    = msg.get("text", "").strip()
    if not chat_id or not text:
        return "", ""
    return chat_id, text


async def send_telegram_reply(chat_id: str, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )


def twilio_twiml(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Message>{escaped}</Message></Response>')

# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"MCP server started — tools: {', '.join(tool_names)}")
            app.state.mcp      = session
            app.state.sessions = SessionManager()
            yield
    # MCP subprocess cleaned up automatically when context exits


app = FastAPI(title="TechFlow WhatsApp Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
async def chat(request: Request):
    """Direct chat API — use for testing without WhatsApp/Telegram."""
    body    = await request.json()
    message = body.get("message", "").strip()
    user_id = body.get("user_id", "api_user")
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    response = await route_message(
        message, user_id, "api",
        request.app.state.sessions, request.app.state.mcp
    )
    return JSONResponse({"response": response, "user_id": user_id})


@app.post("/webhook/twilio")
async def webhook_twilio(request: Request):
    """WhatsApp via Twilio — receives form-encoded POST."""
    form      = await request.form()
    form_dict = dict(form)
    sender, message = parse_twilio(form_dict)
    if not sender or not message:
        return Response(twilio_twiml("Sorry, I didn't receive your message."),
                        media_type="application/xml")
    response = await route_message(
        message, sender, "whatsapp",
        request.app.state.sessions, request.app.state.mcp
    )
    return Response(twilio_twiml(response), media_type="application/xml")


@app.post("/webhook/telegram")
async def webhook_telegram(request: Request):
    """Telegram Bot webhook — receives JSON POST from Telegram servers."""
    body             = await request.json()
    chat_id, message = parse_telegram(body)
    if not chat_id or not message:
        return JSONResponse({"ok": True})
    response = await route_message(
        message, chat_id, "telegram",
        request.app.state.sessions, request.app.state.mcp
    )
    await send_telegram_reply(chat_id, response)
    return JSONResponse({"ok": True})

# ══════════════════════════════════════════════════════════════════════════════
# CLI Demo
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_PROMPTS = [
    "What's the difference between the Pro and Basic plans?",
    "How do I reset my password?",
    "What are the API rate limits for the Pro plan?",
    "Check my order ORD-003",
    "Remind me to upgrade before my trial expires",
    "Do you integrate with Slack?",
]


async def cli_demo():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
    )
    print(f"\n{'='*65}")
    print(f" Project 15 — WhatsApp Agent (CLI Demo)   [{MODEL}]")
    print(f"{'='*65}")
    print("No messaging credentials detected — running CLI demo.")
    print("Set TWILIO_* or TELEGRAM_BOT_TOKEN to enable messaging.\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP tools: {', '.join(t.name for t in tools.tools)}\n")
            print("Endpoints when run as server:")
            print("  POST /chat             — direct API test")
            print("  POST /webhook/twilio   — WhatsApp via Twilio")
            print("  POST /webhook/telegram — Telegram Bot")
            print(f"\n{'─'*65}")
            print("Running demo conversation...\n")

            sessions = SessionManager()
            user_id  = "demo_user"

            for msg in _DEMO_PROMPTS:
                print(f"You:   {msg}")
                response = await route_message(msg, user_id, "cli", sessions, session)
                print(f"Agent: {response}\n")

            print(f"{'─'*65}")
            print("Interactive mode (type 'quit' to exit):\n")

            while True:
                try:
                    msg = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not msg:
                    continue
                if msg.lower() == "quit":
                    break
                response = await route_message(msg, user_id, "cli", sessions, session)
                print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    import uvicorn

    has_twilio   = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
    has_telegram = bool(TELEGRAM_BOT_TOKEN)

    if has_twilio or has_telegram:
        platforms = []
        if has_twilio:   platforms.append("WhatsApp (Twilio)")
        if has_telegram: platforms.append("Telegram")
        print(f"Platforms active: {', '.join(platforms)}")
        print("Expose with: ngrok http 8000")
        uvicorn.run("solution:app", host="0.0.0.0", port=8000, reload=False)
    else:
        asyncio.run(cli_demo())
