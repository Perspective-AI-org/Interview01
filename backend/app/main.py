"""FastAPI application exposing the localrag engine.

Endpoints:
    GET  /api/health          - liveness + backend mode
    GET  /api/stats           - embedder, chunk count, sources, ollama status
    POST /api/ingest/text     - ingest raw text
    POST /api/ingest/file     - ingest an uploaded file (md/txt/pdf)
    POST /api/ask             - ask a question, get a cited answer
    DELETE /api/sources/{name}- remove a source and its chunks

If a production frontend build exists at ``frontend/dist`` it is served at ``/``.
The module imports cleanly with zero optional deps installed, which is what the
CI smoke test relies on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .engine import RagEngine
from .generation import ollama_available
from .schemas import (
    AskRequest,
    AskResponse,
    CitationModel,
    IngestResponse,
    IngestTextRequest,
    SourceInfo,
    StatsResponse,
)

app = FastAPI(
    title="localrag",
    version="0.1.0",
    description="Fully offline document Q&A with local LLMs + citations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One engine for the app's lifetime. Created lazily so importing the module
# (e.g. in CI) never triggers a model download or DB write until first use.
_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    eng = get_engine()
    s = eng.stats()
    return StatsResponse(
        embedder=s["embedder"],
        embed_dim=s["embed_dim"],
        n_chunks=s["n_chunks"],
        ollama_available=ollama_available(),
        sources=[SourceInfo(**src) for src in s["sources"]],
    )


@app.post("/api/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    eng = get_engine()
    try:
        result = eng.ingest_text(req.text, req.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        source=result.source,
        n_chunks=result.n_chunks,
        document_id=result.document_id,
    )


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    eng = get_engine()
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = eng.ingest_file(tmp_path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    # Preserve the original filename as the source label.
    return IngestResponse(
        source=file.filename or result.source,
        n_chunks=result.n_chunks,
        document_id=result.document_id,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    eng = get_engine()
    ans = eng.answer(req.question, top_k=req.top_k, source=req.source)
    return AskResponse(
        answer=ans.answer,
        mode=ans.mode,
        model=ans.model,
        citations=[
            CitationModel(
                marker=c.marker,
                source=c.source,
                chunk_index=c.chunk_index,
                score=c.score,
                snippet=c.snippet,
            )
            for c in ans.citations
        ],
    )


@app.delete("/api/sources/{name}")
def delete_source(name: str) -> JSONResponse:
    eng = get_engine()
    removed = eng.store.delete_source(name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"no such source: {name}")
    return JSONResponse({"deleted": name, "documents_removed": removed})


# --------------------------------------------------------------------- static
# Serve the built React app if present (frontend/dist). Mounted last so it does
# not shadow the /api routes.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=bool(settings.disable_ollama) is False and False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
