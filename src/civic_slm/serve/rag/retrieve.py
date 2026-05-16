"""Top-k retrieval over a built RAG index.

Cosine over normalized BGE embeddings. The index is small (a few hundred
to a few thousand chunks per jurisdiction); doing the matmul in numpy
takes microseconds and saves us a dependency tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from civic_slm.eval.embeddings import DEFAULT_BGE_MODEL
from civic_slm.serve.rag.index import IndexRecord, load_index


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from `top_k`. Carries score so the caller can drop low
    matches or display confidence to the user."""

    record: IndexRecord
    score: float


# Module-level encoder cache: SentenceTransformer load is ~1.5GB and 2-3s.
# Without this, `rag serve` would reload the model on every HTTP request.
# `Any` mirrors the lazy-import pattern in eval/embeddings.py.
_ENCODER_CACHE: dict[str, Any] = {}


def _embed_query(query: str, *, model_id: str) -> np.ndarray:
    """Embed a query with the same encoder used to build the index.

    Lazy import: `sentence-transformers` is in the `eval` extra. Callers
    that haven't installed it get an actionable error rather than the
    `ImportError` deep inside numpy.
    """
    encoder = _ENCODER_CACHE.get(model_id)
    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "civic-slm rag requires the `eval` extra: `uv sync --extra eval`."
            ) from exc
        encoder = SentenceTransformer(model_id)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        _ENCODER_CACHE[model_id] = encoder
    emb = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return np.asarray(emb, dtype=np.float16)[0]


def top_k(
    query: str,
    *,
    index_dir: Path,
    k: int = 4,
    model_id: str = DEFAULT_BGE_MODEL,
) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query`, ranked by cosine similarity.

    `index_dir` is what `build_index` wrote to (`artifacts/<slug>-rag/`).
    `k` is the count of chunks to return; the answer layer typically wants
    4 — enough for citation diversity, few enough to keep prompt context
    under the model's window.
    """
    arr, records = load_index(index_dir)
    if arr.size == 0:
        return []
    q = _embed_query(query, model_id=model_id)
    # arr is already normalized at build time → dot product == cosine.
    scores = arr.astype(np.float32) @ q.astype(np.float32)
    k = min(k, len(records))
    # argpartition then sort the slice — O(N) vs full-sort O(N log N).
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [RetrievedChunk(record=records[i], score=float(scores[i])) for i in top_idx]


def format_context(results: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a `Context:` block ready to prepend to a chat prompt.

    The model is trained on civic data with `Context:` framing already (see
    the synth prompts under `synth/prompts/`), so reusing the convention
    keeps inference behavior close to what it saw during training.
    """
    if not results:
        return "Context: (no relevant passages found in your jurisdiction's corpus)"
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        rec = r.record
        # Carry the citation key into the context so the model can echo it
        # in its answer. Citations look like [1] / [2] / [3] etc.
        meeting = rec.meeting_date or "unknown"
        source = rec.source_url or rec.doc_id
        blocks.append(f"[{i}] (meeting={meeting}, source={source})\n{rec.text.strip()}")
    return "Context:\n" + "\n\n".join(blocks)
