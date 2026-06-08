# Project 26 — Multi-Modal Document Intelligence Agent

> **Stack**: pymupdf4llm · LlamaParse · Whisper · ChromaDB · LiteLLM (GPT-4V / Gemini)  
> **Phase 7 — Advanced Production** | Priority: P0 🔴

---

## What You'll Build

A document intelligence pipeline that processes PDFs as humans do — reading text, understanding charts, interpreting tables, and transcribing audio — then answers questions using evidence from all modalities.

```
Input: Annual report PDF (text + charts + scanned signature pages)
       Board meeting audio recording (MP3)
            ↓
Text extraction   →  text chunk embeddings
Chart analysis    →  "Revenue grew 23% YoY per bar chart"
Table extraction  →  structured JSON from scanned tables
Audio transcript  →  "CEO mentioned Q2 target at 8:23..."
            ↓
Multi-modal vector store (ChromaDB, separate collections)
            ↓
Query: "What did the CEO say about Q2 targets?"
       → retrieves audio transcript chunk (score: 0.94)
       → retrieves revenue chart analysis (score: 0.89)
            ↓
Answer grounded in text + visual + audio evidence
```

---

## Why Text-Only RAG Fails

**Annual report**: 40% of key data is in charts. Text-only RAG misses all of it.  
**Scanned contracts**: 100% image — zero text extracted without vision.  
**Meeting notes**: Audio → transcript needed. Text files don't capture tone or timing.

---

## Milestones

### Milestone 1 — PDF Text Extraction with Layout
Use `pymupdf4llm` to extract markdown from a PDF while preserving: table structure, heading hierarchy, page numbers. Chunk into ~500-char segments, maintaining section context.

### Milestone 2 — Chart and Image Analysis
Extract images from the PDF. For each image, use GPT-4V to: identify the image type (chart/table/diagram/photo), extract structured data as JSON (chart → data points, table → row dicts), generate a text description for embedding.

### Milestone 3 — Scanned PDF Fallback
Detect image-only PDFs (no text layer). Fall back to rendering each page as PNG and OCR-ing via vision LLM. Verify the transcribed text matches the original content.

### Milestone 4 — Audio Transcription Pipeline
Use Whisper to transcribe an audio recording. Chunk transcript by time window (60s segments). Store segments with start/end timestamps as metadata.

### Milestone 5 — Multi-Modal Vector Store
Set up ChromaDB with separate collections: `text_chunks`, `image_descriptions`, `audio_segments`. Store embeddings for all three. Verify all collections are queryable with the same semantic search.

### Milestone 6 — Multi-Modal QA Agent
Build a QA function that:
1. Queries all three collections
2. Includes relevant images directly in the LLM message (as base64 image_url blocks)
3. Returns an answer with typed citations: `[TEXT p.12]`, `[IMAGE chart-revenue]`, `[AUDIO 8:23]`

### Milestone 7 — Accuracy Evaluation
Create a test set of 20 questions with known answers. 10 answerable from text only, 5 from images only, 5 from audio. Compare multi-modal pipeline vs text-only RAG. Target: ≥ 80% accuracy on image/audio questions.

---

## Setup

```bash
cd projects/project26_multimodal_agent
pip install pymupdf4llm chromadb sentence-transformers litellm openai-whisper \
            pillow pydantic fastapi uvicorn python-dotenv

# For LlamaParse (optional, higher quality):
pip install llama-parse
# Get API key at https://cloud.llamaindex.ai
```

---

## Expected Output

```
=== Multi-Modal Pipeline Results ===

Processing: annual_report_2025.pdf
  Text: 47 chunks across 32 pages
  Images: 12 found (8 charts, 3 tables, 1 org chart)
  Audio: board_meeting.mp3 → 94-min transcript, 47 segments

Query: "What are the Q3 revenue figures?"
  Text hit [p.14, score=0.91]: "Revenue for Q3 2025: $12.4M..."
  Chart hit [score=0.89]: "Bar chart shows Q3=$12.4M, up 23% from Q2=$10.1M"
  Answer: "Q3 2025 revenue was $12.4M, a 23% increase from Q2..."
  Sources: [TEXT p.14], [IMAGE revenue-bar-chart]

Accuracy Evaluation:
  Text-only questions: 9/10 correct (90%)
  Image-only questions: 4/5 correct (80%) ← text-only RAG: 0/5
  Audio questions: 4/5 correct (80%)       ← text-only RAG: 0/5
  Overall: 17/20 (85%)
```
