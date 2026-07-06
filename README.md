# localrag

**Fully offline document Q&A with local LLMs + citations.**

localrag is a small, self-hostable retrieval-augmented generation (RAG) engine.
Point it at your PDFs and Markdown notes, and ask questions in plain language.
It retrieves the most relevant passages, answers with **inline citations**, and
runs **100% locally** — no API keys, no cloud, no data leaving your machine.

It is built to degrade gracefully at every layer, so it *always works*:

| Layer | Preferred (if available) | Zero-dependency fallback |
| --- | --- | --- |
| Embeddings | `sentence-transformers` (Hugging Face) | Deterministic **hashing vectorizer** (pure stdlib) |
| Generation | Local **Ollama** LLM (`llama3.2`, `qwen2.5`, …) | **Extractive** answer built from the retrieved passages |
| Documents | PDF via `pypdf` | Markdown / plain text (always) |
| Vector store | — | **SQLite** (always) |

That means you get a genuinely useful, citation-backed answer even on a locked-
down machine with nothing but Python installed — and a much richer experience
once you add a local embedding model and an Ollama server.

---

## Why

Most "chat with your docs" tools require an OpenAI key, ship your documents to a
third party, or collapse into an unusable error the moment a dependency or model
is missing. localrag is the opposite: **offline-first, dependency-optional, and
always answerable.** The hashing-vectorizer + extractive-answer path makes the
whole pipeline hermetic and testable with zero network — which is exactly how
the test suite runs in CI.

---

## Features

- **Ingest** PDFs, Markdown, and plain text (upload, paste, CLI, or API).
- **Smart chunking** with paragraph-aware windows and overlap so citations read
  cleanly and context is never cut mid-thought.
- **Local embeddings** via sentence-transformers, with a deterministic
  feature-hashing fallback that needs no model download and no numpy.
- **SQLite vector store** — portable, single-file, no extra services.
- **Cited answers**: every response carries numbered `[1]`, `[2]` markers that
  map to the exact source passages, with similarity scores and snippets.
- **Local generation** through Ollama, with an automatic extractive fallback.
- **React + Vite + TypeScript** chat UI with document management.
- **CLI** for ingest / ask / stats / serve.
- **Hermetic tests** — the full retrieval pipeline is covered by pytest with no
  network access.

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
  documents ──▶ │  chunking  ──▶  embeddings  ──▶  SQLite store │
  (md/pdf/txt)  └──────────────────────────────────────────────┘
                                                    │
  question ─────────────────▶  embed query  ──▶  cosine search (top-k)
                                                    │
                                             retrieved chunks + scores
                                                    │
                                     ┌──────────────┴───────────────┐
                                     ▼                              ▼
                            Ollama (local LLM)            extractive answer
                                     └──────────────┬───────────────┘
                                                    ▼
                                          answer + citations
```

- `backend/app/chunking.py` — document loading + paragraph-aware chunking
- `backend/app/embeddings.py` — sentence-transformers with hashing fallback
- `backend/app/store.py` — SQLite-backed vector store + cosine search
- `backend/app/generation.py` — Ollama client + extractive answerer
- `backend/app/engine.py` — orchestrates the full RAG pipeline
- `backend/app/main.py` — FastAPI app (also serves the built frontend)
- `backend/app/cli.py` — command-line interface
- `frontend/` — React + Vite + TypeScript chat UI

---

## Tech stack

**Backend:** Python, FastAPI, Pydantic, SQLite, sentence-transformers /
Hugging Face (optional), Ollama (optional), pypdf (optional).
**Frontend:** React, TypeScript, Vite.
**CI:** GitHub Actions.

> The concepts here (chunk → embed → retrieve → generate with citations) are the
> same building blocks that frameworks like **LangChain** expose; localrag
> implements a focused, dependency-light version of that pipeline so it runs
> fully offline. Swapping in LangChain's `RecursiveCharacterTextSplitter` or
> retrievers is a drop-in exercise if you want the larger ecosystem.

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# (optional, for real semantic embeddings + PDF support)
pip install -r requirements-optional.txt

# run the API (also serves the built frontend at http://localhost:8000)
uvicorn app.main:app --reload
```

Ingest and ask from the CLI:

```bash
python -m app.cli ingest ../sample_docs/*.md
python -m app.cli ask "How efficient are commercial solar panels?"
python -m app.cli stats
```

### 2. Frontend (dev mode with hot reload)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

For production, `npm run build` emits `frontend/dist`, which the FastAPI app
serves automatically at `/`.

### 3. (Optional) Local LLM via Ollama

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
ollama serve            # localrag auto-detects it at http://localhost:11434
```

With Ollama running, answers are generated by the model; without it, localrag
returns a grounded extractive answer. Either way you get citations.

---

## Configuration

All settings are environment variables (sensible defaults, zero config needed):

| Variable | Default | Description |
| --- | --- | --- |
| `LOCALRAG_DB` | `localrag.db` | SQLite path (`:memory:` for ephemeral) |
| `LOCALRAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF embedding model |
| `LOCALRAG_HASH_DIM` | `512` | Fallback hashing-vectorizer dimension |
| `LOCALRAG_CHUNK_SIZE` | `900` | Chunk size (characters) |
| `LOCALRAG_CHUNK_OVERLAP` | `150` | Overlap between chunks (characters) |
| `LOCALRAG_TOP_K` | `4` | Passages retrieved per question |
| `LOCALRAG_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `LOCALRAG_OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `LOCALRAG_DISABLE_OLLAMA` | `false` | Force the extractive fallback |

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/stats` | Embedder, chunk count, sources, Ollama status |
| `POST` | `/api/ingest/text` | Ingest raw text `{text, source}` |
| `POST` | `/api/ingest/file` | Ingest an uploaded file (md/txt/pdf) |
| `POST` | `/api/ask` | Ask a question `{question, top_k?, source?}` |
| `DELETE` | `/api/sources/{name}` | Remove a source and its chunks |

Example:

```bash
curl -X POST localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How efficient are solar panels?"}'
```

```json
{
  "answer": "Commercial silicon panels reach 18 to 22 percent efficiency. [1]",
  "mode": "extractive",
  "model": "",
  "citations": [
    { "marker": 1, "source": "solar_power.md", "chunk_index": 1,
      "score": 0.61, "snippet": "Commercial silicon panels typically achieve…" }
  ]
}
```

---

## Testing

The retrieval pipeline is fully covered offline:

```bash
cd backend
pip install -r requirements-dev.txt python-multipart
pytest -q
```

Tests use the deterministic hashing embedder and an in-memory SQLite DB, so they
never touch the network or download a model. CI runs them on Python 3.11 and
3.12 and also builds the frontend.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 xj16.
