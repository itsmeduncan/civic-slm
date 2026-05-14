# pyright: reportPrivateUsage=false
"""Tests for `civic-slm rag` — local single-jurisdiction retrieval-augmented inference.

What's under test: the chunk→manifest join logic, index build/load
round-trip, top_k ranking semantics, and the `Context:` block format.
The sentence-transformers encoder is mocked so tests are fast and don't
download the BGE model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from civic_slm.schema import DocType
from civic_slm.serve.rag.index import (
    IndexRecord,
    _join_chunks_to_manifest,
    build_index,
    load_index,
)
from civic_slm.serve.rag.retrieve import format_context, top_k


def _stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch, dim: int = 4) -> None:
    """Install a fake `sentence_transformers` module that yields deterministic
    embeddings without a network round-trip or a 1.5GB model download."""

    class _FakeEncoder:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def encode(
            self,
            texts: list[str],
            *,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
        ) -> np.ndarray:
            # Per-TEXT deterministic seed (not per-list) — otherwise the
            # vector for "beta" inside encode(["alpha","beta","gamma"]) is
            # different from encode(["beta"]) and cosine becomes random.
            arr = np.zeros((len(texts), dim), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(
                    seed=int.from_bytes(t.encode("utf-8")[:8].ljust(8, b"\0"), "little")
                )
                arr[i] = rng.standard_normal(dim).astype(np.float32)
            if normalize_embeddings:
                arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            return arr

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)


def test_join_chunks_to_manifest_attaches_citation_metadata() -> None:
    """A chunk with a `source_doc_hash` matching a manifest entry should
    pull in source_url + meeting_date."""
    from datetime import UTC, datetime

    from civic_slm.schema import CivicDocument, DocumentChunk

    sha = "a" * 64
    doc = CivicDocument(
        id="ca/test/aaa",
        jurisdiction="test",
        state="CA",
        doc_type=DocType.AGENDA,
        source_url="https://example.gov/a.pdf",  # type: ignore[arg-type]
        retrieved_at=datetime(2026, 5, 14, tzinfo=UTC),
        sha256=sha,
        raw_path="raw/ca/test/aaa.bin",
        text="...",
    )
    chunk = DocumentChunk(
        doc_id="ca/test/aaa",
        chunk_idx=0,
        text="hello world",
        token_count=2,
        source_doc_hash=sha,
    )
    records = _join_chunks_to_manifest([chunk], [doc])
    assert len(records) == 1
    r = records[0]
    assert r.chunk_id == "ca/test/aaa#0"
    assert r.source_url == "https://example.gov/a.pdf"
    assert r.meeting_date == "2026-05-14"
    assert r.sha256 == sha


def test_join_chunks_without_parent_returns_orphan_record() -> None:
    """A chunk with no matching manifest entry shouldn't crash — it
    becomes a row with empty citation fields so the answer layer can
    still reference it by chunk_id."""
    from civic_slm.schema import DocumentChunk

    chunk = DocumentChunk(
        doc_id="ca/test/zzz",
        chunk_idx=0,
        text="orphan",
        token_count=1,
        source_doc_hash="f" * 64,  # valid hex but no matching parent
    )
    records = _join_chunks_to_manifest([chunk], [])
    assert records[0].source_url == ""
    assert records[0].meeting_date is None


def test_build_and_load_index_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip: build → load yields the same row count and embedding shape."""
    _stub_sentence_transformers(monkeypatch)

    # Synthesize a minimal processed/<slug>.jsonl on disk; build_index reads it.
    from civic_slm.ingest.processed import save_chunks
    from civic_slm.schema import DocumentChunk

    chunks = [
        DocumentChunk(
            doc_id=f"ca/tmpville/{i}",
            chunk_idx=0,
            text=f"chunk {i}",
            token_count=2,
            source_doc_hash=f"{i:0>64x}",
        )
        for i in range(3)
    ]
    save_chunks("tmpville", chunks, data_dir=tmp_path)
    # Empty manifest is OK — chunks become orphan records.
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "manifest.jsonl").write_text("", encoding="utf-8")

    out_dir = tmp_path / "tmpville-rag"
    build_index("tmpville", data_dir=tmp_path, out_dir=out_dir)

    arr, records = load_index(out_dir)
    assert arr.shape == (3, 4)
    assert [r.chunk_id for r in records] == [
        "ca/tmpville/0#0",
        "ca/tmpville/1#0",
        "ca/tmpville/2#0",
    ]


def test_top_k_orders_by_similarity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunk whose text matches the query should rank first."""
    _stub_sentence_transformers(monkeypatch)
    from civic_slm.ingest.processed import save_chunks
    from civic_slm.schema import DocumentChunk

    chunks = [
        DocumentChunk(
            doc_id=f"ca/tmpville/{label}",
            chunk_idx=0,
            text=label,
            token_count=1,
            source_doc_hash=None,
        )
        for label in ("alpha", "beta", "gamma")
    ]
    save_chunks("tmpville", chunks, data_dir=tmp_path)
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "manifest.jsonl").write_text("", encoding="utf-8")
    out_dir = tmp_path / "tmpville-rag"
    build_index("tmpville", data_dir=tmp_path, out_dir=out_dir)

    # The fake encoder maps "beta" → a specific vector; querying with
    # "beta" should return the "beta" chunk first (highest cosine).
    results = top_k("beta", index_dir=out_dir, k=3)
    assert len(results) == 3
    assert results[0].record.text == "beta"
    # Scores in descending order.
    assert results[0].score >= results[1].score >= results[2].score


def test_format_context_renders_citation_tags() -> None:
    """The Context: block must include [N] tags the model can echo back."""
    from civic_slm.serve.rag.retrieve import RetrievedChunk

    record = IndexRecord(
        chunk_id="x#0",
        doc_id="x",
        chunk_idx=0,
        text="hello world",
        source_url="https://example.gov/a.pdf",
        meeting_date="2026-05-14",
        sha256="a" * 64,
    )
    block = format_context([RetrievedChunk(record=record, score=0.9)])
    assert block.startswith("Context:")
    assert "[1]" in block
    assert "https://example.gov/a.pdf" in block
    assert "hello world" in block


def test_format_context_handles_empty_retrieval() -> None:
    """No results → an honest 'no relevant passages' note, not a crash."""
    block = format_context([])
    assert "no relevant passages" in block


def test_top_k_raises_when_index_missing(tmp_path: Path) -> None:
    """Asking against an unbuilt index should error clearly, not silently."""
    with pytest.raises(FileNotFoundError, match="not found"):
        top_k("anything", index_dir=tmp_path / "does-not-exist", k=4)
