# Week 14 Resources — Multi-Modal Agents + Advanced Guardrails

---

## 🏁 Start Here (Read in This Order)

1. **OpenAI — Vision Guide** (GPT-4o image input, base64, pricing):
   https://platform.openai.com/docs/guides/vision

2. **NeMo Guardrails Quickstart** (10-min working example):
   https://docs.nvidia.com/nemo/guardrails/latest/getting-started/hello-world.html

3. **Microsoft Presidio** — open-source PII detection (concepts overview):
   https://microsoft.github.io/presidio/

---

## PDF Extraction

### pymupdf4llm
- **GitHub**:
  https://github.com/pymupdf/RAG
- **Docs & Examples**:
  https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/

### LlamaParse
- **Docs**:
  https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/
- **LlamaCloud API** (where you get a key):
  https://cloud.llamaindex.ai
- **LlamaParse GitHub**:
  https://github.com/run-llama/llama_parse

### Azure Document Intelligence
- **Overview**:
  https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview
- **Python SDK Quickstart**:
  https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api?view=doc-intel-4.0.0&pivots=programming-language-python

### Comparison Article
- **PDF Extraction Libraries Benchmark** (Reducto, LlamaParse, pymupdf compared):
  https://reducto.ai/blog/pdf-extraction-benchmarks

---

## Vision LLMs and Multi-Modal Embeddings

### Model Docs
- **OpenAI GPT-4o Vision**:
  https://platform.openai.com/docs/guides/vision

- **Gemini Vision (Google AI)**:
  https://ai.google.dev/gemini-api/docs/vision

- **Anthropic Claude — Vision**:
  https://docs.anthropic.com/en/docs/build-with-claude/vision

### CLIP — Multi-Modal Embeddings
- **CLIP Paper** (Learning Transferable Visual Models From Natural Language Supervision):
  https://arxiv.org/abs/2103.00020

- **OpenCLIP GitHub** (open-source CLIP implementation used for image embeddings):
  https://github.com/mlfoundations/open_clip

- **CLIP + ChromaDB tutorial** (multi-modal vector search):
  https://docs.trychroma.com/guides/multimodal

### Multi-Modal RAG
- **LlamaIndex Multi-Modal Guide**:
  https://docs.llamaindex.ai/en/stable/examples/multi_modal/multi_modal_retrieval/

- **"Multi-Modal RAG" blog post (Aman Chadha)**:
  https://aman.ai/primers/ai/RAG/#multimodal-rag

---

## Whisper — Speech to Text

### Official Resources
- **Whisper Paper** (Robust Speech Recognition via Large-Scale Weak Supervision):
  https://arxiv.org/abs/2212.04356

- **OpenAI Whisper GitHub**:
  https://github.com/openai/whisper

- **OpenAI Whisper API Docs** (cloud transcription endpoint):
  https://platform.openai.com/docs/guides/speech-to-text

### Optimised Local Serving
- **Faster-Whisper** (4× faster via CTranslate2):
  https://github.com/SYSTRAN/faster-whisper

- **WhisperX** (adds speaker diarisation + word-level timestamps):
  https://github.com/m-bain/whisperX

### Real-Time Streaming
- **RealtimeSTT** (real-time Whisper for agent voice input):
  https://github.com/KoljaB/RealtimeSTT

---

## Prompt Injection

### Understanding the Attack
- **OWASP Top 10 for LLMs — LLM01: Prompt Injection**:
  https://owasp.org/www-project-top-10-for-large-language-model-applications/

- **"Prompt Injection Attacks Against GPT-3"** (original academic paper):
  https://arxiv.org/abs/2302.12173

- **Promptmap** — prompt injection testing tool:
  https://github.com/utkusen/promptmap

### Defences
- **"Defending Against Prompt Injection"** (Simon Willison's analysis):
  https://simonwillison.net/2023/Apr/25/dual-llm-pattern/

- **LLM Vulnerabilities overview** — comprehensive attack/defence taxonomy:
  https://learnprompting.org/docs/prompt_hacking/offensive_measures/overview

---

## Llama Guard

### Official Resources
- **Llama Guard 3 Paper**:
  https://arxiv.org/abs/2312.06674

- **Meta's Llama Guard Model Card** (HuggingFace):
  https://huggingface.co/meta-llama/Llama-Guard-3-8B

- **Meta's Purple Llama Project** (the broader safety framework):
  https://github.com/meta-llama/PurpleLlama

### MLCommons AI Safety Taxonomy (what Llama Guard is trained on):
  https://mlcommons.org/2024/04/mlc-aisafety-v0-5/

### Tutorial
- **"Adding Llama Guard to Your LLM Application"** (practical how-to):
  https://ai.meta.com/blog/meta-llama-3-1/

---

## NeMo Guardrails

### Official Resources
- **NeMo Guardrails Docs** (canonical reference):
  https://docs.nvidia.com/nemo/guardrails/latest/

- **NeMo Guardrails GitHub**:
  https://github.com/NVIDIA/NeMo-Guardrails

- **Colang Language Reference** (the policy language):
  https://docs.nvidia.com/nemo/guardrails/latest/colang-language-syntax-guide.html

- **NeMo Guardrails Examples** (topical rails, grounding rails, moderation):
  https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples

### Tutorials
- **"Guardrails for LLMs in Production"** (NVIDIA Developer Blog):
  https://developer.nvidia.com/blog/safeguarding-llms-with-guardrails/

---

## PII Detection and Anonymisation

### Microsoft Presidio
- **Presidio GitHub**:
  https://github.com/microsoft/presidio

- **Presidio Docs** (recognisers, anonymisers, API):
  https://microsoft.github.io/presidio/

- **Presidio — Custom Recogniser Guide**:
  https://microsoft.github.io/presidio/analyzer/adding_recognizers/

### spaCy NER (used in exercises)
- **spaCy NER Docs**:
  https://spacy.io/usage/linguistic-features#named-entities

- **spaCy Models** (en_core_web_sm, en_core_web_trf):
  https://spacy.io/usage/models

### Guardrails AI
- **Guardrails AI Docs** (declarative output validators):
  https://www.guardrailsai.com/docs

- **Guardrails AI GitHub**:
  https://github.com/guardrails-ai/guardrails

### GDPR and Privacy
- **GDPR Article 4** — legal definition of PII:
  https://gdpr.eu/article-4-definitions/

- **"Privacy by Design for AI Systems"** (practical principles):
  https://www.ipc.on.ca/wp-content/uploads/resources/pbd-implement-7found-principles.pdf

---

## Videos and Courses

- **DeepLearning.AI — "Building Multimodal Search and RAG"** (free short course):
  https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/

- **Andrej Karpathy — "Deep Dive into LLMs"** (includes vision model architecture):
  https://youtu.be/7xTGNNLPyMI

- **NVIDIA — NeMo Guardrails Tutorial Video**:
  https://www.youtube.com/watch?v=2Gm0M-9vd4g

---

## Tools Checklist

| Tool | Purpose | Install |
|---|---|---|
| `pymupdf4llm` | PDF → Markdown extraction | `pip install pymupdf4llm` |
| `llama-parse` | Advanced PDF extraction (cloud) | `pip install llama-parse` |
| `openai` | GPT-4o vision API | `pip install openai` |
| `openai-whisper` | Local speech-to-text | `pip install openai-whisper` |
| `faster-whisper` | Faster local Whisper | `pip install faster-whisper` |
| `nemoguardrails` | Programmable safety rails | `pip install nemoguardrails` |
| `presidio-analyzer` | PII detection | `pip install presidio-analyzer` |
| `presidio-anonymizer` | PII anonymisation | `pip install presidio-anonymizer` |
| `guardrails-ai` | Output validation framework | `pip install guardrails-ai` |
| `open_clip_torch` | CLIP image embeddings | `pip install open-clip-torch` |
