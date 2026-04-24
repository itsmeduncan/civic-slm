"""Unit tests for the section-aware chunker. PDF extraction itself is tested
behind a fixture; here we exercise the pure-text path."""

from __future__ import annotations

from civic_slm.ingest.pdf import chunk_text


def _para(words: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(words))


def test_chunker_emits_at_least_one_chunk_for_short_doc() -> None:
    text = "Intro paragraph one.\n\nIntro paragraph two."
    chunks = chunk_text("d1", text)
    assert len(chunks) == 1
    assert chunks[0].chunk_idx == 0
    assert chunks[0].token_count > 0
    assert chunks[0].section_path == []


def test_chunker_respects_target_size_and_overlaps() -> None:
    paras = "\n\n".join(_para(300, prefix=f"p{i}_") for i in range(5))
    chunks = chunk_text("d1", paras, target_tokens=500, overlap_tokens=100)
    assert len(chunks) >= 2
    # Sequential chunks have monotonic indices.
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    # All chunks are non-empty.
    assert all(c.token_count > 0 for c in chunks)


def test_chunker_tracks_section_headings() -> None:
    text = (
        "LAND USE ELEMENT\n\n"
        "First paragraph of land use.\n\n"
        "Second paragraph of land use.\n\n"
        "HOUSING ELEMENT\n\n"
        "Housing paragraph one."
    )
    chunks = chunk_text("d1", text, target_tokens=10000)
    assert len(chunks) == 1
    assert "LAND USE ELEMENT" in chunks[0].section_path
    assert "HOUSING ELEMENT" in chunks[0].section_path
