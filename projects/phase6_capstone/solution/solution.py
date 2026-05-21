"""
SOLUTION — Project 6 (Option C): Multi-Agent Content Pipeline
Input: topic + brand voice document → blog post + social posts + email newsletter.
3 specialized agents: Researcher → Writer → Editor

Run:
    python solution.py "the future of wearable AI" --brand brand_voice.md
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import asyncio
import json
import re
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()


# ── Data Models ────────────────────────────────────────────────────────────────

class ResearchFindings(BaseModel):
    topic: str
    key_facts: list[str]
    angles: list[str]
    target_audience: str
    tone_notes: str


class ContentDraft(BaseModel):
    blog_post: str         # 600-800 words
    twitter_thread: list[str]  # 5 tweets
    linkedin_post: str
    email_subject: str
    email_body: str


class EditedContent(BaseModel):
    blog_post: str
    twitter_thread: list[str]
    linkedin_post: str
    email_subject: str
    email_body: str
    edit_notes: str


# ── Agent Helper ───────────────────────────────────────────────────────────────

async def call_llm(system: str, user: str, max_tokens: int = 2048) -> str:
    r = await achat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
    return get_text(r)


# ── Agent 1: Researcher ────────────────────────────────────────────────────────

async def researcher_agent(topic: str, brand_voice: str) -> ResearchFindings:
    print("🔍 Researcher working...")
    system = (
        "You are a content research specialist. Given a topic and brand voice document, "
        "identify key facts, interesting angles, target audience, and tone guidance. "
        "Return ONLY valid JSON matching the schema. No markdown."
    )
    schema = """{
  "topic": "str",
  "key_facts": ["fact1", "fact2", "fact3", "fact4", "fact5"],
  "angles": ["unique angle 1", "unique angle 2", "unique angle 3"],
  "target_audience": "description of who this is for",
  "tone_notes": "specific guidance on tone/style from the brand voice"
}"""
    raw = await call_llm(system,
        f"Topic: {topic}\n\nBrand Voice:\n{brand_voice}\n\nSchema:\n{schema}", max_tokens=1024)
    s = raw.find("{"); e = raw.rfind("}") + 1
    data = json.loads(raw[s:e])
    return ResearchFindings(**data)


# ── Agent 2: Writer ────────────────────────────────────────────────────────────

async def writer_agent(research: ResearchFindings, brand_voice: str) -> ContentDraft:
    print("✍ Writer working...")

    blog_task = call_llm(
        f"You are a content writer. Write a 600-800 word blog post. Brand voice: {brand_voice[:500]}",
        f"Topic: {research.topic}\nKey facts: {research.key_facts}\nAngle: {research.angles[0]}\n"
        f"Audience: {research.target_audience}\nTone: {research.tone_notes}",
        max_tokens=1200,
    )

    twitter_task = call_llm(
        "You are a social media expert. Write a Twitter/X thread of exactly 5 tweets. "
        "Return JSON array of 5 strings. No markdown, no numbering.",
        f"Topic: {research.topic}\nKey points: {research.key_facts[:3]}\nAudience: {research.target_audience}",
        max_tokens=600,
    )

    linkedin_task = call_llm(
        "Write a professional LinkedIn post (150-200 words) with a hook, value, and CTA.",
        f"Topic: {research.topic}\nAngle: {research.angles[1] if len(research.angles) > 1 else research.angles[0]}",
        max_tokens=400,
    )

    email_task = call_llm(
        "Write an email newsletter. Return JSON: {\"subject\": \"...\", \"body\": \"...\"}. No markdown.",
        f"Topic: {research.topic}\nKey facts: {research.key_facts}\nTone: {research.tone_notes}",
        max_tokens=800,
    )

    blog, twitter_raw, linkedin, email_raw = await asyncio.gather(
        blog_task, twitter_task, linkedin_task, email_task
    )

    # Parse twitter thread
    try:
        ts = twitter_raw.find("["); te = twitter_raw.rfind("]") + 1
        tweets = json.loads(twitter_raw[ts:te])
    except Exception:
        tweets = [t.strip() for t in twitter_raw.split("\n") if t.strip()][:5]

    # Parse email
    try:
        es = email_raw.find("{"); ee = email_raw.rfind("}") + 1
        email_data = json.loads(email_raw[es:ee])
    except Exception:
        email_data = {"subject": f"The Future of {research.topic}", "body": email_raw}

    return ContentDraft(
        blog_post=blog,
        twitter_thread=tweets,
        linkedin_post=linkedin,
        email_subject=email_data["subject"],
        email_body=email_data["body"],
    )


# ── Agent 3: Editor ────────────────────────────────────────────────────────────

async def editor_agent(draft: ContentDraft, research: ResearchFindings, brand_voice: str) -> EditedContent:
    print("📝 Editor working...")
    system = (
        "You are a senior editor. Review and improve all content for consistency, "
        "brand voice alignment, grammar, and engagement. Make targeted improvements. "
        "Return JSON: {\"blog_post\": ..., \"twitter_thread\": [...], "
        "\"linkedin_post\": ..., \"email_subject\": ..., \"email_body\": ..., \"edit_notes\": ...}"
    )
    prompt = (
        f"Brand voice: {brand_voice[:400]}\n"
        f"Topic: {research.topic}\n"
        f"Audience: {research.target_audience}\n\n"
        f"DRAFT CONTENT:\n"
        f"BLOG:\n{draft.blog_post[:800]}...\n\n"
        f"TWITTER: {json.dumps(draft.twitter_thread)}\n\n"
        f"LINKEDIN:\n{draft.linkedin_post}\n\n"
        f"EMAIL SUBJECT: {draft.email_subject}\n"
        f"EMAIL:\n{draft.email_body[:400]}..."
    )
    raw = await call_llm(system, prompt, max_tokens=3000)
    s = raw.find("{"); e = raw.rfind("}") + 1
    data = json.loads(raw[s:e])
    return EditedContent(**data)


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

async def run_content_pipeline(topic: str, brand_voice_path: str = None) -> EditedContent:
    brand_voice = ""
    if brand_voice_path and Path(brand_voice_path).exists():
        brand_voice = Path(brand_voice_path).read_text()
    else:
        brand_voice = (
            "Professional yet approachable. Use clear language, avoid jargon. "
            "Lead with insights, back with data. End with actionable takeaways. "
            "Audience: tech-savvy professionals aged 25-40."
        )

    print(f"\n🚀 Content Pipeline: {topic}")
    print("=" * 50)

    # Step 1: Research (sequential, others depend on it)
    research = await researcher_agent(topic, brand_voice)
    print(f"   ✅ Research done — {len(research.key_facts)} facts, {len(research.angles)} angles")

    # Step 2: Writing (parallel possible, but using sequential for clarity)
    draft = await writer_agent(research, brand_voice)
    print(f"   ✅ Drafts done — blog: {len(draft.blog_post.split())} words")

    # Step 3: Editing
    final = await editor_agent(draft, research, brand_voice)
    print(f"   ✅ Editing done")
    print(f"   📝 Notes: {final.edit_notes[:80]}")

    return final


def save_outputs(content: EditedContent, topic: str):
    slug = re.sub(r"[^\w\s]", "", topic).replace(" ", "_").lower()[:30]
    out_dir = Path(f"output_{slug}")
    out_dir.mkdir(exist_ok=True)

    (out_dir / "blog_post.md").write_text(content.blog_post)
    (out_dir / "twitter_thread.txt").write_text("\n\n".join(
        f"[{i+1}/5] {t}" for i, t in enumerate(content.twitter_thread)
    ))
    (out_dir / "linkedin_post.txt").write_text(content.linkedin_post)
    (out_dir / "email.txt").write_text(f"Subject: {content.email_subject}\n\n{content.email_body}")
    (out_dir / "edit_notes.txt").write_text(content.edit_notes)

    print(f"\n📁 All content saved to: {out_dir}/")
    print(f"   - blog_post.md")
    print(f"   - twitter_thread.txt ({len(content.twitter_thread)} tweets)")
    print(f"   - linkedin_post.txt")
    print(f"   - email.txt")


async def main():
    topic = " ".join(sys.argv[1:]).split("--brand")[0].strip() if sys.argv[1:] else "the future of wearable AI"
    brand_path = None
    if "--brand" in sys.argv:
        idx = sys.argv.index("--brand")
        brand_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    content = await run_content_pipeline(topic, brand_path)
    save_outputs(content, topic)


if __name__ == "__main__":
    asyncio.run(main())
