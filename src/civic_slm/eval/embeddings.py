"""Sentence-similarity helpers used by the factuality scorer.

The default factuality scorer (`scorers._word_overlap`) is fast and has no
dependencies, but rewards verbatim copying and penalizes correct paraphrase
(see `MODEL_CARD.md` "Known limitations of the eval harness"). This module
provides the BGE-based replacement called for in `CLAUDE.md` and tracked as
v0.2.x Track A1 in `ROADMAP.md`.

Usage:

    from civic_slm.eval.embeddings import bge_similarity_fn
    sim_fn = bge_similarity_fn()
    score = sim_fn("gold answer", "prediction")  # in [0.0, 1.0]

The similarity function is shaped like `scorers.score_factuality`'s optional
`similarity_fn` kwarg — pass it through and the scorer uses it instead of
word-overlap.

The model is loaded lazily on first call so that:

  - `civic-slm eval run --similarity word_overlap` (the default) does **not**
    pull in `sentence-transformers` or download a multi-hundred-megabyte
    model;
  - tests that don't exercise the BGE path don't pay the import cost.

The first call downloads + loads the model (~1.5GB for the default
`BAAI/bge-large-en-v1.5`). Subsequent calls reuse the cached encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_BGE_MODEL = "BAAI/bge-large-en-v1.5"

_ENCODER_CACHE: dict[str, object] = {}


def bge_similarity_fn(model_id: str = DEFAULT_BGE_MODEL) -> Callable[[str, str], float]:
    """Return a `(gold, prediction) -> [0.0, 1.0]` cosine-similarity callable.

    The encoder is a `sentence-transformers` dual-encoder. Cosine similarity
    on normalized embeddings is in `[-1, 1]`; we map it to `[0, 1]` via
    `(s + 1) / 2` so it composes cleanly with the citation-match score in
    `score_factuality`.

    Why dual-encoder, not the cross-encoder reranker: the reranker emits a
    relevance score (logit-shaped) that is harder to bound and depends on
    asymmetric query/passage roles. A dual-encoder cosine is the standard
    "semantic similarity" measurement and matches what an reviewer expects
    when they read "BGE similarity" in the model card.
    """
    cached = _ENCODER_CACHE.get(model_id)
    if cached is None:
        # Lazy import — sentence-transformers is in the `eval` extra only.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        cached = SentenceTransformer(model_id)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
        _ENCODER_CACHE[model_id] = cached
    encoder = cached

    def _sim(gold: str, prediction: str) -> float:
        if not gold.strip() or not prediction.strip():
            return 0.0
        # Normalize to unit length; `util.cos_sim` returns a 1x1 tensor.
        from sentence_transformers import util  # type: ignore[import-not-found]

        embs = encoder.encode([gold, prediction], normalize_embeddings=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        cos: float = float(util.cos_sim(embs[0], embs[1]).item())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        # Map cosine [-1, 1] → [0, 1].
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))

    return _sim


def reset_encoder_cache() -> None:
    """Drop cached encoders. Used by tests to avoid cross-test contamination."""
    _ENCODER_CACHE.clear()
