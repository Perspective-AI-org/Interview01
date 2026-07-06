"""SQLite-backed document + vector store.

Vectors are stored as JSON arrays alongside their chunk text and provenance.
For a local, single-user Q&A tool this is simpler and more portable than a
dedicated vector DB, and a brute-force cosine scan over a few thousand chunks is
comfortably fast. The store is intentionally dependency-free (stdlib ``sqlite3``
+ ``json``) so it works everywhere Python does.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .chunking import Chunk
from .embeddings import cosine_similarity


@dataclass
class ScoredChunk:
    """A retrieved chunk plus its similarity score and DB id."""

    id: int
    text: str
    source: str
    chunk_index: int
    score: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    embed_dim   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


class VectorStore:
    """A thread-safe SQLite store for chunks and their embeddings."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # check_same_thread=False + an explicit lock lets FastAPI's threadpool
        # share one connection safely.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ writes
    def add_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Persist a document's chunks with their embeddings. Returns doc id."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            raise ValueError("cannot add an empty document")

        source = chunks[0].source
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (source, n_chunks) VALUES (?, ?)",
                (source, len(chunks)),
            )
            doc_id = int(cur.lastrowid)
            rows = [
                (
                    doc_id,
                    c.source,
                    c.chunk_index,
                    c.char_start,
                    c.char_end,
                    c.text,
                    json.dumps([round(float(x), 6) for x in emb]),
                    len(emb),
                )
                for c, emb in zip(chunks, embeddings)
            ]
            self._conn.executemany(
                """
                INSERT INTO chunks
                    (document_id, source, chunk_index, char_start, char_end,
                     text, embedding, embed_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        return doc_id

    def delete_source(self, source: str) -> int:
        """Delete all documents/chunks for a given source filename."""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM documents WHERE source = ?", (source,)
            )
            self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            self._conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM chunks; DELETE FROM documents;"
            )
            self._conn.commit()

    # ------------------------------------------------------------------- reads
    def count_chunks(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def list_sources(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT source,
                       SUM(n_chunks) AS n_chunks,
                       MIN(created_at) AS created_at
                FROM documents
                GROUP BY source
                ORDER BY MIN(created_at) DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def _iter_embeddings(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, source, chunk_index, text, embedding FROM chunks"
            ).fetchall()
        for r in rows:
            yield r, json.loads(r["embedding"])

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
        source: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Brute-force cosine search over stored chunks.

        Optionally restrict to a single ``source``. Returns the ``top_k``
        highest-scoring chunks in descending score order.
        """

        scored: List[ScoredChunk] = []
        for row, emb in self._iter_embeddings():
            if source is not None and row["source"] != source:
                continue
            score = cosine_similarity(query_embedding, emb)
            scored.append(
                ScoredChunk(
                    id=int(row["id"]),
                    text=row["text"],
                    source=row["source"],
                    chunk_index=int(row["chunk_index"]),
                    score=float(score),
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: max(0, top_k)]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
