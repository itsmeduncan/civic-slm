"""End-to-end ingest tests using a fake recipe + in-memory fetcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from civic_slm.ingest import manifest
from civic_slm.ingest.harness import DiscoveredDoc, crawl
from civic_slm.schema import DocType


@dataclass
class _FakeRecipe:
    jurisdiction: str = "test-city"
    state: str = "CA"
    instruction: str = ""
    docs: list[DiscoveredDoc] = ()  # type: ignore[assignment]

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        return list(self.docs)


@pytest.mark.asyncio
async def test_crawl_lands_new_docs_and_skips_dupes(tmp_path: Path) -> None:
    docs = [
        DiscoveredDoc(
            title="Council Meeting 2026-01-15",
            source_url="https://example.gov/agenda-1.txt",
            doc_type=DocType.AGENDA,
            meeting_date="2026-01-15",
        ),
        DiscoveredDoc(
            title="Council Meeting 2026-02-15",
            source_url="https://example.gov/agenda-2.txt",
            doc_type=DocType.AGENDA,
            meeting_date="2026-02-15",
        ),
    ]
    recipe = _FakeRecipe(docs=docs)

    contents = {
        "https://example.gov/agenda-1.txt": b"Item 1. Approve minutes from prior meeting.",
        "https://example.gov/agenda-2.txt": b"Item 1. Budget amendment.\n\nItem 2. Public comment.",
    }

    async def fake_fetch(url: str) -> bytes:
        return contents[url]

    landed = await crawl(
        recipe=recipe, data_dir=tmp_path, since="2026-01-01", max_docs=10, fetch=fake_fetch
    )
    assert len(landed) == 2
    assert all(d.text for d in landed)

    # Manifest persisted.
    rows = manifest.load_manifest(tmp_path)
    assert len(rows) == 2

    # Re-running is idempotent.
    landed2 = await crawl(
        recipe=recipe, data_dir=tmp_path, since="2026-01-01", max_docs=10, fetch=fake_fetch
    )
    assert landed2 == []
    assert len(manifest.load_manifest(tmp_path)) == 2


def test_sniff_suffix_pdf_magic_overrides_extensionless_url() -> None:
    """Regression for #65: efiles.portlandoregon.gov serves PDFs from URLs
    like `/Record/<id>/file/document` with no extension. Trusting the URL
    alone produced `.bin` files, which `_extract_text` then read as text —
    polluting the manifest with raw PDF byte streams.
    """
    from civic_slm.ingest.harness import _sniff_suffix  # pyright: ignore[reportPrivateUsage]

    url = "https://efiles.portlandoregon.gov/Record/16934535/file/document"
    assert _sniff_suffix(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", url) == ".pdf"
    # Non-PDF extensionless URL still falls back to .bin (signals "unknown").
    assert _sniff_suffix(b"plain text payload", url) == ".bin"
    # URL extension is still honored when there's no magic-byte signal.
    assert _sniff_suffix(b"some text", "https://example.gov/agenda.txt") == ".txt"


def test_extract_text_uses_pdf_extractor_when_suffix_is_wrong(tmp_path: Path) -> None:
    """`_extract_text` must magic-byte-sniff, not trust the suffix. A `.bin`
    file whose contents start with `%PDF-` should still route through pypdf
    instead of `read_text`. See #65.

    Gated on pypdf because it lives in the optional `ingest` extra, which CI's
    lint/type job doesn't sync (the production path imports pypdf lazily for
    the same reason).
    """
    PdfWriter = pytest.importorskip("pypdf").PdfWriter

    from civic_slm.ingest.harness import _extract_text  # pyright: ignore[reportPrivateUsage]

    target = tmp_path / "agenda.bin"  # intentionally wrong suffix
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with target.open("wb") as fh:
        writer.write(fh)
    assert target.read_bytes().startswith(b"%PDF-")

    text = _extract_text(target)
    # The blank page has no extractable text — but crucially, the function
    # MUST NOT have returned the raw PDF byte stream.
    assert "%PDF-" not in text
