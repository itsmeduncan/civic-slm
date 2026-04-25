"""Tests for the BGE similarity helper.

The full test downloads ~1.5GB and is gated behind the
`CIVIC_SLM_RUN_BGE_TEST` env var so the default test suite stays fast and
network-free. The lightweight test runs against a stubbed encoder via
`sentence_transformers`'s test cache when available; otherwise it asserts
the import-error path is actionable.
"""

from __future__ import annotations

import os

import pytest
import typer

from civic_slm.eval import embeddings
from civic_slm.eval.runner import _resolve_similarity  # pyright: ignore[reportPrivateUsage]


def test_resolves_similarity_returns_none_for_word_overlap() -> None:
    """`--similarity word_overlap` must keep the scorer test-isolated."""
    assert _resolve_similarity("word_overlap", "anything") is None


def test_resolves_similarity_rejects_unknown() -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_similarity("cosine", "anything")


def test_reset_encoder_cache_drops_state() -> None:
    embeddings._ENCODER_CACHE["sentinel"] = object()
    embeddings.reset_encoder_cache()
    assert "sentinel" not in embeddings._ENCODER_CACHE


@pytest.mark.skipif(
    not os.environ.get("CIVIC_SLM_RUN_BGE_TEST"),
    reason="downloads ~1.5GB; set CIVIC_SLM_RUN_BGE_TEST=1 to run.",
)
def test_bge_similarity_ranks_paraphrase_above_unrelated() -> None:
    sim = embeddings.bge_similarity_fn()
    gold = "The City Council approved Conditional Use Permit CUP 24-031 by a 5-0 vote."
    paraphrase = "The council voted 5-0 to grant CUP 24-031."
    unrelated = "Public Works will sweep streets every Tuesday next quarter."
    s_para = sim(gold, paraphrase)
    s_unr = sim(gold, unrelated)
    assert 0.0 <= s_unr <= s_para <= 1.0
    # A real dual-encoder should put the paraphrase a comfortable margin above
    # the unrelated sentence on this kind of civic content.
    assert s_para - s_unr >= 0.05, f"sim_para={s_para:.3f} vs sim_unr={s_unr:.3f}"


def test_bge_similarity_handles_empty_inputs() -> None:
    """Empty strings short-circuit to 0.0 without loading the encoder."""
    # Stash a stub in the cache so we don't actually load the model.
    embeddings._ENCODER_CACHE["stub"] = object()
    sim = embeddings.bge_similarity_fn("stub")
    assert sim("", "anything") == 0.0
    assert sim("anything", "  ") == 0.0
    embeddings.reset_encoder_cache()
