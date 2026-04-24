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
