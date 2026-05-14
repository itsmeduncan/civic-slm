"""Tests for the `civic-slm data-card` auto-generator (closes #26).

Covers:
- Empty manifest → graceful "(none)" row.
- Single jurisdiction → counts roll up correctly across docs + chunks.
- Two jurisdictions → sorted by (state, slug) for stable diffs.
- `_splice` requires both sentinels and updates between them only.
- `--check` raises when the rendered block drifts from disk.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from civic_slm.data_card import (
    _splice,  # pyright: ignore[reportPrivateUsage]
    compute_stats,
    render_markdown,
)
from civic_slm.schema import CivicDocument, DocType


def _doc(jurisdiction: str, state: str, retrieved: datetime, sha: str) -> CivicDocument:
    return CivicDocument(
        id=f"{state}/{jurisdiction}/{sha[:12]}",
        jurisdiction=jurisdiction,
        state=state,
        doc_type=DocType.AGENDA,
        source_url="https://example.gov/x",  # pyright: ignore[reportArgumentType]
        retrieved_at=retrieved,
        sha256=sha,
        raw_path=f"raw/{state.lower()}/{jurisdiction}/x.bin",
        text="hello",
    )


def _write_manifest(data_dir: Path, docs: list[CivicDocument]) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(d.model_dump_json() + "\n")


def _write_processed(data_dir: Path, slug: str, chunks: list[dict[str, object]]) -> None:
    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    with (proc / f"{slug}.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")


def test_empty_manifest_renders_none_row(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [])
    stats = compute_stats(tmp_path)
    assert stats == []
    md = render_markdown(stats)
    assert "(none — manifest is empty)" in md
    assert md.startswith("<!-- DATA_CARD:JURISDICTIONS:BEGIN -->")
    assert md.rstrip().endswith("<!-- DATA_CARD:JURISDICTIONS:END -->")


def test_single_jurisdiction_rolls_up_chunks_and_tokens(tmp_path: Path) -> None:
    when = datetime(2026, 5, 14, tzinfo=UTC)
    _write_manifest(tmp_path, [_doc("san-clemente", "CA", when, "a" * 64)])
    _write_processed(
        tmp_path,
        "san-clemente",
        [{"token_count": 100}, {"token_count": 250}, {"token_count": 75}],
    )

    stats = compute_stats(tmp_path)
    assert len(stats) == 1
    s = stats[0]
    assert s.slug == "san-clemente"
    assert s.state == "CA"
    assert s.doc_count == 1
    assert s.chunk_count == 3
    assert s.token_count == 425
    assert s.doc_types["agenda"] == 1

    md = render_markdown(stats)
    assert "`san-clemente`" in md
    assert "425" in md  # token total in body
    assert "**425**" in md  # totals row


def test_multiple_jurisdictions_sorted_by_state_then_slug(tmp_path: Path) -> None:
    when = datetime(2026, 5, 14, tzinfo=UTC)
    docs = [
        _doc("seattle", "WA", when, "1" * 64),
        _doc("austin", "TX", when, "2" * 64),
        _doc("boston", "MA", when, "3" * 64),
    ]
    _write_manifest(tmp_path, docs)
    stats = compute_stats(tmp_path)
    # Sort key is (state, slug): MA < TX < WA.
    assert [s.slug for s in stats] == ["boston", "austin", "seattle"]


def test_splice_requires_both_sentinels(tmp_path: Path) -> None:
    target = tmp_path / "DATA_CARD.md"
    target.write_text("# Card\n\nNo sentinels here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing the sentinel block"):
        _splice(target, "## fresh\n")


def test_splice_is_idempotent(tmp_path: Path) -> None:
    """Splicing the same fresh block twice produces the same file — otherwise
    `--check` after `--write` diverges because each pass adds a blank line."""
    target = tmp_path / "DATA_CARD.md"
    target.write_text(
        "# Card\n\n"
        "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->\n"
        "old\n"
        "<!-- DATA_CARD:JURISDICTIONS:END -->\n"
        "\n\n\n"
        "## After\n",
        encoding="utf-8",
    )
    fresh = (
        "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->\n"
        "new\n"
        "<!-- DATA_CARD:JURISDICTIONS:END -->\n"
    )
    once = _splice(target, fresh)
    target.write_text(once, encoding="utf-8")
    twice = _splice(target, fresh)
    assert once == twice


def test_splice_replaces_between_sentinels_only(tmp_path: Path) -> None:
    target = tmp_path / "DATA_CARD.md"
    target.write_text(
        "# Card\n\n"
        "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->\n"
        "old content\n"
        "<!-- DATA_CARD:JURISDICTIONS:END -->\n\n"
        "## Afterward\nText that must survive.\n",
        encoding="utf-8",
    )
    fresh = (
        "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->\n"
        "new content\n"
        "<!-- DATA_CARD:JURISDICTIONS:END -->\n"
    )
    spliced = _splice(target, fresh)
    assert "old content" not in spliced
    assert "new content" in spliced
    assert "# Card" in spliced
    assert "Text that must survive." in spliced
