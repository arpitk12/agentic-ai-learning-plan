"""
Project 15 — WhatsApp Multi-Agent Assistant: MCP + RAG (Starter)

Build a conversational AI reachable from your phone via WhatsApp or Telegram.
This file contains the FastAPI app, multi-agent routing, session management,
and messaging platform webhooks. The MCP server in mcp_server.py provides
the RAG and lookup tools as structured MCP tool calls.

Architecture:
  Phone → WhatsApp/Telegram → POST /webhook/* → SessionManager →
  classify_intent() → specialist agent → MCP tool call → LLM → reply

Run modes:
  python starter.py                                    ← CLI demo (no credentials needed)
  uvicorn starter:app --host 0.0.0.0 --port 8000      ← full API server

Required env vars (all optional — app works without them in CLI mode):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER
  TELEGRAM_BOT_TOKEN
"""

import os, sys, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Form
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

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

# ══════════════════════════════════════════════════════════════════════════════
# Session Manager — per-user conversation history
# ══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """Keeps the last MAX_TURNS exchanges per user (keyed by phone number / chat ID)."""

    MAX_TURNS = 8    # number of user+assistant turn pairs to keep

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}

    def get_history(self, user_id: str) -> list[dict]:
        """Return message history for user_id (may be empty list)."""
        return self._sessions.get(user_id, [])

    def add_message(self, user_id: str, role: str, content: str):
        """Append a message, trimming to MAX_TURNS * 2 messages."""
        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append({"role": role, "content": content})
        cap = self.MAX_TURNS * 2
        if len(self._sessions[user_id]) > cap:
            self._sessions[user_id] = self._sessions[user_id][-cap:]

    def clear(self, user_id: str):
        self._sessions.pop(user_id, None)

    def history_as_text(self, user_id: str) -> str:
        """Return last 6 messages as plain text for the intent classifier."""
        msgs = self.get_history(user_id)[-6:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


# ══════════════════════════════════════════════════════════════════════════════
# LLM Helper
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are TechFlow's helpful AI assistant, reachable via WhatsApp and Telegram. "
    "Be concise (phone-friendly: ≤ 200 words), warm, and accurate. "
    "Use plain text — no markdown since messaging apps render it poorly."
)


async def _llm(messages: list[dict], system: str = _SYSTEM) -> str:
    r = await achat(messages, system=system, max_tokens=350)
    return get_text(r)


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool Helper
# ══════════════════════════════════════════════════════════════════════════════

async def call_mcp_tool(mcp_session: ClientSession, tool: str, args: dict) -> str:
    """Call a named MCP tool and return the text content of the result."""
    result = await mcp_session.call_tool(tool, args)
    if result.content:
        return result.content[0].text
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

INTENT_CATEGORIES = ["rag", "product", "order", "reminder", "general"]


async def classify_intent(message: str, history_text: str) -> str:
    """TODO: Send message + history_text to LLM with a prompt that defines
       all 5 intent categories. Return one of INTENT_CATEGORIES.
       Default to 'general' if the reply is not in the list.

       Categories:
         rag      — knowledge base questions (features, pricing, how-to, policies)
         product  — specific product comparison / plan recommendation
         order    — order or subscription status (message contains ORD-NNN or mentions 'my order')
         reminder — user wants to set a reminder or note
         general  — anything else (greetings, off-topic, unclear)
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Specialist Agents
# ══════════════════════════════════════════════════════════════════════════════

async def rag_agent(message: str, history: list[dict],
                    mcp_session: ClientSession) -> str:
    """TODO: Knowledge base RAG agent.
       1. call_mcp_tool(mcp_session, 'search_knowledge_base', {'query': message})
       2. Build prompt: include retrieved chunks + message + last 4 history messages
       3. Call LLM and return response
    """
    raise NotImplementedError


async def product_agent(message: str, history: list[dict],
                        mcp_session: ClientSession) -> str:
    """TODO: Product info agent.
       1. Extract product name from message (look for 'pro', 'basic', 'enterprise',
          or use 'techflow pro' as default if unclear)
       2. call_mcp_tool(mcp_session, 'get_product_info', {'product_name': name})
       3. Build prompt with product info + message, call LLM, return response
    """
    raise NotImplementedError


async def order_agent(message: str, history: list[dict],
                      mcp_session: ClientSession) -> str:
    """TODO: Order status agent.
       1. Extract order ID from message with re.findall(r'ORD-\\d+', message, re.I)
       2. If found: call_mcp_tool(mcp_session, 'get_order_status', {'order_id': order_ids[0]})
       3. If not found: ask user to provide their order ID
       4. Build prompt with order info, call LLM, return response
    """
    raise NotImplementedError


async def reminder_agent(message: str, user_id: str,
                         mcp_session: ClientSession) -> str:
    """TODO: Reminder creation agent.
       1. Extract the reminder text from the message
          (strip phrases like 'remind me to', 'set a reminder', etc.)
       2. call_mcp_tool(mcp_session, 'create_reminder',
                        {'user_id': user_id, 'text': reminder_text})
       3. Return the confirmation string from the tool directly (no LLM needed)
    """
    raise NotImplementedError


async def general_agent(message: str, history: list[dict]) -> str:
    """TODO: General fallback agent.
       Build messages: history + current user message.
       Call LLM with _SYSTEM prompt and return response.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

async def route_message(message: str, user_id: str, platform: str,
                        sessions: SessionManager,
                        mcp_session: ClientSession) -> str:
    """TODO: Full message routing pipeline.
       1. Handle special commands: 'clear' → sessions.clear(user_id), 'help' → help text
       2. sessions.add_message(user_id, 'user', message)
       3. intent = await classify_intent(message, sessions.history_as_text(user_id))
       4. history = sessions.get_history(user_id)
       5. Dispatch to specialist agent based on intent
       6. sessions.add_message(user_id, 'assistant', response)
       7. Print debug line: f"[{user_id[:8]}] intent={intent}  platform={platform}"
       8. Return response
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Platform Parsers
# ══════════════════════════════════════════════════════════════════════════════

def parse_twilio(form_data: dict) -> tuple[str, str]:
    """TODO: Extract (sender_id, message_text) from Twilio webhook form data.
       sender_id = form_data.get('From', '')       e.g. 'whatsapp:+1234567890'
       message   = form_data.get('Body', '').strip()
       Return ('', '') if either is missing.
    """
    raise NotImplementedError


def parse_telegram(body: dict) -> tuple[str, str]:
    """TODO: Extract (chat_id, message_text) from Telegram webhook JSON body.
       chat_id = str(body.get('message', {}).get('chat', {}).get('id', ''))
       message = body.get('message', {}).get('text', '').strip()
       Return ('', '') if either is missing.
    """
    raise NotImplementedError


async def send_telegram_reply(chat_id: str, text: str):
    """Send a reply via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )


def twilio_twiml(text: str) -> str:
    """Wrap reply text in Twilio TwiML response envelope."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escaped}</Message></Response>'


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App — Lifespan + Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """TODO: Start MCP server subprocess and maintain a persistent client session.

       Steps:
         1. server_params = StdioServerParameters(
                command=sys.executable,
                args=[str(MCP_SERVER_PATH)],
            )
         2. async with stdio_client(server_params) as (read, write):
         3.     async with ClientSession(read, write) as session:
         4.         await session.initialize()
         5.         tools = await session.list_tools()
         6.         print tool names so students can see them
         7.         app.state.mcp      = session
         8.         app.state.sessions = SessionManager()
         9.         yield   ← app runs here; cleanup on shutdown
    """
    raise NotImplementedError


app = FastAPI(title="TechFlow WhatsApp Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
async def chat(request: Request):
    """Direct chat endpoint — for testing without WhatsApp/Telegram."""
    body    = await request.json()
    message = body.get("message", "").strip()
    user_id = body.get("user_id", "api_user")
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    response = await route_message(
        message, user_id, "api", request.app.state.sessions, request.app.state.mcp
    )
    return JSONResponse({"response": response, "user_id": user_id})


@app.post("/webhook/twilio")
async def webhook_twilio(request: Request):
    """WhatsApp webhook — Twilio posts form data here."""
    form     = await request.form()
    form_dict = dict(form)
    sender, message = parse_twilio(form_dict)
    if not sender or not message:
        return Response(twilio_twiml("Sorry, I didn't receive your message."),
                        media_type="application/xml")
    response = await route_message(
        message, sender, "whatsapp", request.app.state.sessions, request.app.state.mcp
    )
    return Response(twilio_twiml(response), media_type="application/xml")


@app.post("/webhook/telegram")
async def webhook_telegram(request: Request):
    """Telegram webhook — Bot API posts JSON here."""
    body      = await request.json()
    chat_id, message = parse_telegram(body)
    if not chat_id or not message:
        return JSONResponse({"ok": True})
    response = await route_message(
        message, chat_id, "telegram", request.app.state.sessions, request.app.state.mcp
    )
    await send_telegram_reply(chat_id, response)
    return JSONResponse({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# CLI Demo — runs without any messaging credentials
# ══════════════════════════════════════════════════════════════════════════════

async def cli_demo():
    """Run an interactive CLI session so the project works without Twilio/Telegram."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
    )
    print(f"\n{'='*65}")
    print(f" Project 15 — WhatsApp Agent (CLI Demo)   [{MODEL}]")
    print(f"{'='*65}")
    print("CLI demo mode — no messaging credentials needed.")
    print("Type 'clear' to reset session, 'quit' to exit.\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP tools ready: {', '.join(t.name for t in tools.tools)}\n")

            sessions = SessionManager()
            user_id  = "cli_user"

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
        # Start as a proper API server (webhooks ready to receive messages)
        print(f"Starting API server — WhatsApp: {'✅' if has_twilio else '❌'}  Telegram: {'✅' if has_telegram else '❌'}")
        print("Set up ngrok: ngrok http 8000  →  copy URL to your Twilio/Telegram webhook settings")
        uvicorn.run("starter:app", host="0.0.0.0", port=8000, reload=False)
    else:
        # No credentials — run CLI demo instead
        asyncio.run(cli_demo())
