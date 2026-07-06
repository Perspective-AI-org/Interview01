"""Answer generation from retrieved context.

Primary path: a **local Ollama** server (``/api/generate``) running any model
you've pulled (llama3.2, qwen2.5, mistral, ...). No API key, no cloud.

Fallback path: when Ollama is unreachable or disabled, an **extractive**
answer is assembled directly from the highest-scoring retrieved chunks. This
guarantees the tool produces a grounded, citation-backed response even with no
LLM present at all — the RAG pipeline degrades gracefully instead of failing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Sequence

from .config import settings
from .store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering strictly from the provided context. "
    "Cite the sources you use with bracketed numbers like [1], [2] that refer to "
    "the numbered context passages. If the answer is not in the context, say you "
    "don't have enough information. Be concise."
)


@dataclass
class Citation:
    """A source passage referenced by an answer."""

    marker: int
    source: str
    chunk_index: int
    score: float
    snippet: str


@dataclass
class Answer:
    """A generated answer plus its provenance."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    mode: str = "extractive"  # "ollama" or "extractive"
    model: str = ""


def build_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""

    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {ch.source})\n{ch.text.strip()}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: Sequence[ScoredChunk]) -> str:
    context = build_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with bracketed citations):"
    )


def _citations_for(chunks: Sequence[ScoredChunk]) -> List[Citation]:
    cites: List[Citation] = []
    for i, ch in enumerate(chunks, start=1):
        snippet = ch.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."
        cites.append(
            Citation(
                marker=i,
                source=ch.source,
                chunk_index=ch.chunk_index,
                score=round(ch.score, 4),
                snippet=snippet,
            )
        )
    return cites


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _top_sentences(text: str, query: str, max_sentences: int = 2) -> List[str]:
    """Pick the sentences from ``text`` most lexically relevant to ``query``."""

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return []
    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_terms:
        return sentences[:max_sentences]

    def overlap(sentence: str) -> int:
        s_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(q_terms & s_terms)

    ranked = sorted(range(len(sentences)), key=lambda i: overlap(sentences[i]), reverse=True)
    chosen = sorted(ranked[:max_sentences])
    return [sentences[i] for i in chosen]


def extractive_answer(question: str, chunks: Sequence[ScoredChunk]) -> Answer:
    """Build a grounded answer from retrieved chunks with no LLM.

    Summarizes each of the top passages down to its most query-relevant
    sentences and attaches an inline citation marker, so the response is
    genuinely useful and fully traceable even without a model.
    """

    citations = _citations_for(chunks)
    if not chunks:
        return Answer(
            answer=(
                "I don't have any indexed documents that match this question. "
                "Try ingesting a relevant document first."
            ),
            citations=[],
            mode="extractive",
        )

    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        sents = _top_sentences(ch.text, question, max_sentences=2)
        if not sents:
            continue
        lines.append(f"{' '.join(sents)} [{i}]")

    body = " ".join(lines) if lines else "See the cited passages below."
    preamble = (
        "Based on the most relevant passages in the indexed documents "
        "(no local LLM was available, so this answer is extracted directly "
        "from the sources):\n\n"
    )
    return Answer(
        answer=preamble + body,
        citations=citations,
        mode="extractive",
    )


def _ollama_generate(prompt: str) -> str:
    """Call the local Ollama /api/generate endpoint. Raises on failure."""

    url = settings.ollama_url.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {"model": settings.ollama_model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=settings.ollama_timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def ollama_available() -> bool:
    """Best-effort probe of the local Ollama server (fast, non-fatal)."""

    if settings.disable_ollama:
        return False
    try:
        url = settings.ollama_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def generate_answer(question: str, chunks: Sequence[ScoredChunk]) -> Answer:
    """Generate an answer, preferring Ollama and falling back to extractive.

    Never raises for a missing/unreachable LLM — the extractive path always
    produces a grounded, cited answer.
    """

    citations = _citations_for(chunks)

    if not settings.disable_ollama and chunks:
        try:
            prompt = build_prompt(question, chunks)
            text = _ollama_generate(prompt)
            return Answer(
                answer=text,
                citations=citations,
                mode="ollama",
                model=settings.ollama_model,
            )
        except Exception:  # noqa: BLE001 - fall back on any Ollama failure
            pass

    return extractive_answer(question, chunks)
