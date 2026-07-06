"""Runtime configuration for localrag.

Everything is environment-driven so the whole stack runs with zero config out of
the box, but can be pointed at a real Ollama server / different embedding model
without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Central settings object, populated from the environment."""

    # Where the SQLite vector store lives. ":memory:" is honoured for tests.
    db_path: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_DB", str(Path(__file__).resolve().parents[2] / "localrag.db")
        )
    )

    # Embedding model name (sentence-transformers). If the library or model is
    # unavailable we transparently fall back to a hashing vectorizer.
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Dimensionality of the deterministic hashing-vectorizer fallback.
    hash_dim: int = field(default_factory=lambda: _env_int("LOCALRAG_HASH_DIM", 512))

    # Chunking parameters (in characters).
    chunk_size: int = field(default_factory=lambda: _env_int("LOCALRAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("LOCALRAG_CHUNK_OVERLAP", 150)
    )

    # Retrieval defaults.
    top_k: int = field(default_factory=lambda: _env_int("LOCALRAG_TOP_K", 4))

    # Ollama generation endpoint. If unreachable we fall back to an extractive
    # answer built directly from the retrieved chunks.
    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "LOCALRAG_OLLAMA_URL", "http://localhost:11434"
        )
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("LOCALRAG_OLLAMA_MODEL", "llama3.2")
    )
    ollama_timeout: int = field(
        default_factory=lambda: _env_int("LOCALRAG_OLLAMA_TIMEOUT", 60)
    )
    # Force the extractive fallback even if Ollama happens to be reachable.
    disable_ollama: bool = field(
        default_factory=lambda: _env_bool("LOCALRAG_DISABLE_OLLAMA", False)
    )


settings = Settings()
