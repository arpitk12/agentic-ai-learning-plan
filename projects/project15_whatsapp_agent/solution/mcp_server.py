"""
Project 15 — MCP Server (Solution)

Full implementation of all 4 MCP tools:
  - search_knowledge_base  (TF-IDF RAG over 15 KB docs)
  - get_product_info       (product catalog lookup)
  - get_order_status       (mock order database)
  - create_reminder        (in-memory reminder store)
"""

import math, re
from collections import Counter
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TechFlow Assistant MCP Server")

# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════

KB_DOCUMENTS = [
    {"id": "kb001", "title": "TechFlow Pro — Overview",
     "content": "TechFlow Pro is our flagship SaaS platform for team collaboration. It includes unlimited projects, 100GB storage, priority support, and advanced analytics dashboard. Perfect for teams of 10-500 members."},
    {"id": "kb002", "title": "TechFlow Basic — Overview",
     "content": "TechFlow Basic is our starter plan for small teams. Includes up to 5 projects, 10GB storage, email support, and basic reporting. Ideal for freelancers and teams under 10 members."},
    {"id": "kb003", "title": "TechFlow Enterprise — Overview",
     "content": "TechFlow Enterprise offers unlimited everything: projects, storage, users, and dedicated account management. Includes SSO, custom integrations, SLA guarantee, and on-premise deployment option. Contact sales for pricing."},
    {"id": "kb004", "title": "Pricing — All Plans",
     "content": "TechFlow Basic: $29/month or $290/year (save 2 months). TechFlow Pro: $99/month or $990/year. TechFlow Enterprise: custom pricing, contact sales@techflow.io. All plans include 14-day free trial. No credit card required for trial."},
    {"id": "kb005", "title": "Password Reset",
     "content": "To reset your password: 1) Go to app.techflow.io 2) Click Forgot Password 3) Enter your email 4) Check inbox for reset link (valid 30 minutes) 5) Set new password. If email not received, check spam folder or contact support@techflow.io."},
    {"id": "kb006", "title": "Two-Factor Authentication",
     "content": "Enable 2FA under Settings → Security → Two-Factor Authentication. We support authenticator apps (Google Authenticator, Authy) and SMS. If you lose 2FA device access, use backup codes saved when you enabled 2FA, or contact support."},
    {"id": "kb007", "title": "API Rate Limits",
     "content": "TechFlow API rate limits: Basic plan: 100 requests/minute, 10,000/day. Pro plan: 1,000 requests/minute, 100,000/day. Enterprise: unlimited. Rate limit headers in all responses: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset."},
    {"id": "kb008", "title": "Integrations",
     "content": "TechFlow integrates with Slack, Microsoft Teams, GitHub, GitLab, Jira, Notion, Zapier, and 50+ more. Go to Settings → Integrations to connect. Pro and Enterprise plans include API access for custom integrations. Webhook support available for real-time events."},
    {"id": "kb009", "title": "Data Export",
     "content": "Export your data anytime from Settings → Data → Export. Supported formats: CSV, JSON, PDF, Excel. Full account export includes all projects, tasks, comments, and files. Processing time: up to 24 hours for large accounts. Email notification when ready."},
    {"id": "kb010", "title": "Billing and Invoices",
     "content": "Billing processed on the 1st of each month for monthly plans. Annual plans billed once per year. View invoices at Settings → Billing → Invoices. We accept Visa, Mastercard, American Express, and PayPal. Update payment method at Settings → Billing → Payment Method."},
    {"id": "kb011", "title": "Cancellation Policy",
     "content": "Cancel anytime from Settings → Account → Cancel Subscription. Monthly plans: cancelled at end of billing period, no partial month refund. Annual plans: pro-rated refund for unused months if cancelled within 30 days. After 30 days, no refund but access continues."},
    {"id": "kb012", "title": "Storage and File Limits",
     "content": "File size limits: Basic 50MB per file, Pro 250MB per file, Enterprise 1GB per file. Total storage: Basic 10GB, Pro 100GB, Enterprise unlimited. Supported types: PDF, DOC, DOCX, JPG, PNG, GIF, XLS, XLSX, CSV, ZIP. Videos on Pro and Enterprise only."},
    {"id": "kb013", "title": "Team Management",
     "content": "Invite team members from Settings → Team → Invite Members. Roles: Owner (full access), Admin (manage team and settings), Member (access projects), Viewer (read-only). Remove members from Settings → Team → Manage. Transfer ownership from Settings → Team → Transfer Ownership."},
    {"id": "kb014", "title": "Mobile App",
     "content": "TechFlow mobile app available for iOS (App Store) and Android (Google Play). Features: view and edit projects, receive push notifications, upload files from camera, offline viewing mode with sync when online. Included with all plans."},
    {"id": "kb015", "title": "Support and SLA",
     "content": "Support by plan: Basic: email support 48-hour response. Pro: email and live chat 4-hour response. Enterprise: email, chat, phone, dedicated account manager, 1-hour SLA. Community forum at community.techflow.io. Status page at status.techflow.io."},
]

PRODUCTS = {
    "techflow pro":        {"name": "TechFlow Pro",        "price": "$99/mo or $990/yr",  "trial": "14 days free",
                            "features": ["Unlimited projects", "100GB storage", "Priority support", "Advanced analytics", "1,000 API req/min"]},
    "techflow basic":      {"name": "TechFlow Basic",      "price": "$29/mo or $290/yr",  "trial": "14 days free",
                            "features": ["5 projects", "10GB storage", "Email support", "Basic reporting", "100 API req/min"]},
    "techflow enterprise": {"name": "TechFlow Enterprise", "price": "Custom — contact sales@techflow.io", "trial": "Custom POC",
                            "features": ["Unlimited everything", "SSO + SAML", "Dedicated account manager", "1-hour SLA", "On-premise option"]},
}

ORDERS = {
    "ORD-001": {"status": "active",    "plan": "TechFlow Pro",   "started": "2026-01-01",
                "next_billing": "2026-07-01", "amount": "$99.00/month"},
    "ORD-002": {"status": "cancelled", "plan": "TechFlow Basic", "started": "2025-11-01",
                "ended": "2026-03-01"},
    "ORD-003": {"status": "trial",     "plan": "TechFlow Pro",   "trial_ends": "2026-06-21",
                "amount": "$0.00 (trial - will charge $99 on 2026-06-21)"},
    "ORD-004": {"status": "past_due",  "plan": "TechFlow Basic", "started": "2026-03-01",
                "amount": "$29.00/month - PAYMENT FAILED"},
}

REMINDERS: list[dict] = []
_REMINDER_COUNTER = {"n": 0}

# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF Index (built at module load)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

_ALL_TEXTS = [f"{d['title']} {d['content']}" for d in KB_DOCUMENTS]
_TOKENISED = [_tokenise(t) for t in _ALL_TEXTS]
_TF_LIST   = [Counter(t) for t in _TOKENISED]
_N         = len(KB_DOCUMENTS)
_DF        = Counter()
for _toks in _TOKENISED:
    for _tok in set(_toks):
        _DF[_tok] += 1
_IDF = {t: math.log((_N + 1) / (_DF[t] + 1)) for t in _DF}   # smoothed IDF


def _tfidf_score(query_tokens: list[str], tf: Counter) -> float:
    return sum(tf.get(t, 0) * _IDF.get(t, 0.0) for t in query_tokens)

# ══════════════════════════════════════════════════════════════════════════════
# MCP Tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the TechFlow knowledge base using TF-IDF retrieval.
    Returns the top 3 most relevant document chunks for a given query.
    """
    q_tokens = _tokenise(query)
    scores   = [(_tfidf_score(q_tokens, tf), i) for i, tf in enumerate(_TF_LIST)]
    top      = sorted(scores, reverse=True)[:3]

    results = [(score, i) for score, i in top if score > 0]
    if not results:
        return "No relevant information found in the knowledge base."

    lines: list[str] = []
    for rank, (score, idx) in enumerate(results, 1):
        doc = KB_DOCUMENTS[idx]
        lines.append(f"{rank}. [{doc['id']}] {doc['title']}")
        lines.append(f"   {doc['content']}")
    return "\n".join(lines)


@mcp.tool()
def get_product_info(product_name: str) -> str:
    """Look up TechFlow product details by name (case-insensitive)."""
    key = product_name.strip().lower()

    # Exact match first
    product = PRODUCTS.get(key)

    # Partial match: 'pro' → 'techflow pro'
    if not product:
        for k, v in PRODUCTS.items():
            if key in k or k in key:
                product = v
                break

    if not product:
        available = ", ".join(PRODUCTS.keys())
        return f"Product '{product_name}' not found. Available products: {available}"

    features_str = "\n  - ".join(product["features"])
    return (
        f"{product['name']}\n"
        f"Price : {product['price']}\n"
        f"Trial : {product['trial']}\n"
        f"Features:\n  - {features_str}"
    )


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Look up subscription/order status by order ID (e.g. 'ORD-001')."""
    key   = order_id.strip().upper()
    order = ORDERS.get(key)

    if not order:
        available = ", ".join(ORDERS.keys())
        return f"Order '{order_id}' not found. Known orders: {available}"

    status = order["status"].replace("_", " ").title()
    lines  = [f"Order {key} — {status}", f"Plan: {order['plan']}"]

    if "next_billing" in order:
        lines.append(f"Next billing: {order['next_billing']} ({order.get('amount', '')})")
    if "ended" in order:
        lines.append(f"Cancelled: {order['ended']}")
    if "trial_ends" in order:
        lines.append(f"Trial ends: {order['trial_ends']} ({order.get('amount', '')})")
    if order["status"] == "past_due":
        lines.append("ACTION REQUIRED: Update payment method at Settings > Billing.")

    return "\n".join(lines)


@mcp.tool()
def create_reminder(user_id: str, text: str) -> str:
    """Create a reminder for a user and return a confirmation string."""
    _REMINDER_COUNTER["n"] += 1
    rid = f"R-{_REMINDER_COUNTER['n']:03d}"
    REMINDERS.append({
        "id":         rid,
        "user_id":    user_id,
        "text":       text,
        "created_at": datetime.now().isoformat(),
    })
    return f"Reminder {rid} set: \"{text}\""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()   # stdio transport — do not print to stdout
