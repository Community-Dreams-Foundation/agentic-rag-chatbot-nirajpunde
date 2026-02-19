# Architecture Overview

## Goal
Provide a brief, readable overview of how your chatbot works:
- ingestion
- indexing
- retrieval + grounding with citations
- memory writing
- optional safe tool execution

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
- **Supported inputs**: `.txt` files only
- **Parsing approach**: `TextLoader` from LangChain (plain text, UTF-8)
- **Chunking strategy**: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- **Metadata captured per chunk**:
  - `source` (filename)
  - `chunk_id` (1-based index)
  - `locator` (e.g. "chunk 2" for citations)

### 2) Indexing / Storage
- **Vector store**: FAISS (local, persistent) — avoids ChromaDB Python 3.14/Pydantic issues
- **Persistence**: `./faiss_index/` directory
- **Embeddings**: Google Gemini `text-embedding-004` via `langchain-google-genai`
- **Optional lexical index (BM25)**: Not implemented; top-k semantic search only

### 3) Retrieval + Grounded Answering
- **Retrieval method**: Top-k similarity search (k=5)
- **How citations are built**: Each retrieved doc → `{source, locator, snippet}`; citations come from retrieved docs, not parsed from LLM output
- **Failure behavior**: If retrieval returns no docs, respond with "I cannot find this in the uploaded documents" and return empty citations (no hallucinated sources)

### 4) Memory System (Selective)
- **What counts as "high-signal" memory**: User role, preferences, workflows; org-wide learnings (interfaces, bottlenecks)
- **What we do NOT store**: Raw transcripts, PII, secrets, trivial chitchat
- **How we decide when to write**: LLM extracts facts with `{should_write, target, summary, confidence}`; only append when confidence ≥ 0.7
- **Format written to**:
  - `USER_MEMORY.md` — user-specific facts
  - `COMPANY_MEMORY.md` — org-wide learnings
  - Format: `- [YYYY-MM-DD] Summary text`

### 5) Optional: Safe Tooling (Open-Meteo)
- Not implemented in this version.

---

## Tradeoffs & Next Steps

**Why this design?**
- LangChain + FAISS + Gradio: minimal setup, good for demos and local development
- Google Gemini: single API for LLM and embeddings, good value
- .txt-only ingestion: simplifies parsing and keeps scope manageable

**What we would improve with more time:**
- Hybrid retrieval (BM25 + embeddings) for better recall
- Reranking for higher precision
-
