"""
Project 6 Starter — Multi-Agent Content Pipeline (Capstone)

Build a three-stage async content factory:

  Researcher ──► Writer ──► Editor
     │               │           │
  ResearchFindings  ContentDraft  EditedContent

Input:   a topic string  +  optional brand_voice.md document
Output:  blog post · Twitter thread · LinkedIn post · email newsletter
         (all saved to output_<slug>/ directory)

Run:
    python starter.py "the future of wearable AI"
    python starter.py "AI in healthcare" --brand brand_voice.md

Concepts practised:
  - Multi-agent pipeline with typed handoffs (Pydantic models)
  - asyncio.gather() for parallelising LLM calls inside a single agent
  - Structured JSON output parsing (find-first-brace trick)
  - Sequential orchestration where each stage consumes the previous output

What you need to implement (TODOs 1-4):
  1. researcher_agent()      — one LLM call → JSON → ResearchFindings
  2. writer_agent()          — 4 parallel LLM calls → parse outputs → ContentDraft
  3. editor_agent()          — one LLM call → JSON → EditedContent
  4. run_content_pipeline()  — await all three agents in order, return final
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

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
    topic:           str
    key_facts:       list[str]   # at least 5 interesting facts
    angles:          list[str]   # at least 3 unique hooks / story angles
    target_audience: str
    tone_notes:      str         # extracted from brand voice


class ContentDraft(BaseModel):
    blog_post:      str          # 600–800 words
    twitter_thread: list[str]   # exactly 5 tweets
    linkedin_post:  str          # 150–200 words
    email_subject:  str
    email_body:     str


class EditedContent(BaseModel):
    blog_post:      str
    twitter_thread: list[str]
    linkedin_post:  str
    email_subject:  str
    email_body:     str
    edit_notes:     str          # editor's summary of changes made


# ── Agent Helper ───────────────────────────────────────────────────────────────

async def call_llm(system: str, user: str, max_tokens: int = 2048) -> str:
    """Thin async wrapper — returns the text content of a single LLM call."""
    r = await achat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
    return get_text(r)


# ── Agent 1: Researcher ────────────────────────────────────────────────────────

async def researcher_agent(topic: str, brand_voice: str) -> ResearchFindings:
    """
    Analyse the topic in the context of the brand voice document and return
    structured research findings as a ResearchFindings object.

    TODO 1 — implement this agent:
      a. Write a system prompt that instructs the model to act as a content
         research specialist and return ONLY valid JSON with no markdown fences.
         The JSON must match ResearchFindings exactly:
             {
               "topic":           "...",
               "key_facts":       ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"],
               "angles":          ["angle 1", "angle 2", "angle 3"],
               "target_audience": "...",
               "tone_notes":      "..."
             }
      b. Build a user message:
             f"Topic: {topic}\\n\\nBrand Voice:\\n{brand_voice}\\n\\nSchema:\\n{schema}"
      c. Call:  raw = await call_llm(system, user_message, max_tokens=1024)
      d. Extract JSON:
             s = raw.find("{");  e = raw.rfind("}") + 1
             data = json.loads(raw[s:e])
      e. Return ResearchFindings(**data)
    """
    print("🔍 Researcher working...")
    # TODO 1: implement researcher agent
    raise NotImplementedError("researcher_agent() not implemented yet")


# ── Agent 2: Writer ────────────────────────────────────────────────────────────

async def writer_agent(research: ResearchFindings, brand_voice: str) -> ContentDraft:
    """
    Produce all 4 content formats in parallel using asyncio.gather().

    TODO 2 — implement this agent:
      a. Create 4 coroutines (do NOT await them yet):
           blog_task     — 600-800 word blog post in brand voice using research.angles[0]
                           System: "You are a content writer. Write a 600-800 word blog post.
                                    Brand voice: {brand_voice[:500]}"
                           User:   topic, key_facts, angle, target_audience, tone_notes
           twitter_task  — 5-tweet thread as a raw JSON array
                           System: "You are a social media expert. Write a Twitter/X thread of
                                    exactly 5 tweets. Return JSON array of 5 strings.
                                    No markdown, no numbering."
                           User:   topic, key_facts[:3], target_audience
           linkedin_task — 150-200 word post with hook, value, and CTA
                           System: "Write a professional LinkedIn post (150-200 words)
                                    with a hook, value, and CTA."
                           User:   topic, angles[1] (or angles[0] if only one)
           email_task    — newsletter as raw JSON {"subject": "...", "body": "..."}
                           System: 'Write an email newsletter. Return JSON:
                                    {"subject": "...", "body": "..."}. No markdown.'
                           User:   topic, key_facts, tone_notes

      b. Await all 4 in parallel:
             blog, twitter_raw, linkedin, email_raw = await asyncio.gather(
                 blog_task, twitter_task, linkedin_task, email_task)

      c. Parse twitter_raw → list[str]  (find '[' ... ']' then json.loads; fall back to split)
         Parse email_raw   → dict       (find '{' ... '}' then json.loads; fall back to default)

      d. Return ContentDraft(blog_post=blog, twitter_thread=tweets,
                             linkedin_post=linkedin,
                             email_subject=email_data["subject"],
                             email_body=email_data["body"])

    Fallback examples:
        tweets = [t.strip() for t in twitter_raw.split("\\n") if t.strip()][:5]
        email_data = {"subject": f"The Future of {research.topic}", "body": email_raw}
    """
    print("✍  Writer working...")
    # TODO 2: implement parallel writer agent
    raise NotImplementedError("writer_agent() not implemented yet")


# ── Agent 3: Editor ────────────────────────────────────────────────────────────

async def editor_agent(draft: ContentDraft, research: ResearchFindings, brand_voice: str) -> EditedContent:
    """
    Polish all content for brand alignment, grammar, and engagement.
    Return improved versions plus a brief edit_notes summary.

    TODO 3 — implement this agent:
      a. Write a system prompt: senior editor role, return ONLY JSON matching
         EditedContent schema (no markdown):
             {
               "blog_post": "...",
               "twitter_thread": ["tweet1", "tweet2", ...],
               "linkedin_post": "...",
               "email_subject": "...",
               "email_body": "...",
               "edit_notes": "brief summary of changes made"
             }
      b. Build a user prompt that includes:
           - brand_voice[:400]
           - research.topic and research.target_audience
           - Draft content:
               BLOG (first 800 chars + "...")
               TWITTER (json.dumps(draft.twitter_thread))
               LINKEDIN (full text)
               EMAIL SUBJECT + body (first 400 chars + "...")
      c. raw = await call_llm(system, prompt, max_tokens=3000)
      d. Extract JSON: s = raw.find("{"); e = raw.rfind("}") + 1
         data = json.loads(raw[s:e])
      e. Return EditedContent(**data)
    """
    print("📝 Editor working...")
    # TODO 3: implement editor agent
    raise NotImplementedError("editor_agent() not implemented yet")


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

async def run_content_pipeline(topic: str, brand_voice_path: str = None) -> EditedContent:
    """
    Orchestrate all three agents sequentially.

    TODO 4 — implement this function:
      a. Load brand voice:
           if brand_voice_path and Path(brand_voice_path).exists():
               brand_voice = Path(brand_voice_path).read_text()
           else:
               brand_voice = <default string below>
      b. Print the pipeline header.
      c. Step 1 — research:
             research = await researcher_agent(topic, brand_voice)
             print(f"   ✅ Research done — {len(research.key_facts)} facts, {len(research.angles)} angles")
      d. Step 2 — write:
             draft = await writer_agent(research, brand_voice)
             print(f"   ✅ Drafts done — blog: {len(draft.blog_post.split())} words")
      e. Step 3 — edit:
             final = await editor_agent(draft, research, brand_voice)
             print(f"   ✅ Editing done")
             print(f"   📝 Notes: {final.edit_notes[:80]}")
      f. Return final
    """
    # Default brand voice used when no file is provided
    default_brand_voice = (
        "Professional yet approachable. Use clear language, avoid jargon. "
        "Lead with insights, back with data. End with actionable takeaways. "
        "Audience: tech-savvy professionals aged 25-40."
    )
    brand_voice = default_brand_voice
    if brand_voice_path and Path(brand_voice_path).exists():
        brand_voice = Path(brand_voice_path).read_text()

    print(f"\n🚀 Content Pipeline: {topic}")
    print("=" * 50)

    # TODO 4: await each agent in sequence and return the final EditedContent
    raise NotImplementedError("run_content_pipeline() not implemented yet")


# ── Output Saver (already complete) ───────────────────────────────────────────

def save_outputs(content: EditedContent, topic: str):
    """Save each content format to a separate file inside output_<slug>/."""
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
    for name in ["blog_post.md", "twitter_thread.txt", "linkedin_post.txt", "email.txt", "edit_notes.txt"]:
        print(f"   - {name}")


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
