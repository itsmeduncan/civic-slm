"""browser-use / browser-harness driver.

Each city has a *recipe* — a high-level natural-language instruction plus a list
of expected fields. The recipe is handed to a browser-use Agent (Anthropic-backed)
which navigates the city site and returns a structured list of document URLs.
The harness then downloads each URL via httpx and writes it to data/raw/.

Why an agentic crawler instead of a hand-written scraper: California city sites
run on a long tail of platforms (Granicus, Legistar, custom WordPress, IQM,
PrimeGov). An agent given "find council agendas for the last 12 months" usually
beats per-platform CSS selectors that rot. We pay for it with API tokens and
slower wallclock. Tradeoff is acceptable at the v0 corpus scale (~thousands of
docs, not millions).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx

from civic_slm.ingest import manifest
from civic_slm.logging import get_logger
from civic_slm.schema import CivicDocument, DocType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = get_logger(__name__)


@dataclass(frozen=True)
class DiscoveredDoc:
    """One document the agent found, before we've fetched the bytes."""

    title: str
    source_url: str
    doc_type: DocType
    meeting_date: str | None = None  # ISO date string when relevant


class Recipe(Protocol):
    """A city recipe — must expose a `city` slug and an async `discover` method."""

    @property
    def city(self) -> str: ...

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]: ...


async def crawl(
    *,
    recipe: Recipe,
    data_dir: Path,
    since: str,
    max_docs: int,
    fetch: Callable[[str], Awaitable[bytes]] | None = None,
) -> list[CivicDocument]:
    """Discover via the recipe, fetch bytes, append new docs to the manifest.

    Skips docs whose sha256 is already in the manifest, so re-runs are idempotent.
    """
    fetcher = fetch or _default_fetch
    discovered = await recipe.discover(since=since, max_docs=max_docs)
    log.info("discovered_docs", city=recipe.city, count=len(discovered))

    seen = manifest.known_hashes(data_dir)
    landed: list[CivicDocument] = []
    for d in discovered:
        try:
            data = await fetcher(d.source_url)
        except Exception as exc:
            log.warning("fetch_failed", url=d.source_url, error=str(exc))
            continue
        sha = manifest.sha256_bytes(data)
        if sha in seen:
            continue

        raw_rel = _raw_path(recipe.city, d, sha)
        raw_abs = data_dir / raw_rel
        raw_abs.parent.mkdir(parents=True, exist_ok=True)
        raw_abs.write_bytes(data)

        text = _extract_text(raw_abs)
        if not text:
            log.warning("empty_extraction", url=d.source_url)
            continue

        doc = CivicDocument(
            id=f"{recipe.city}/{sha[:12]}",
            city=recipe.city,
            doc_type=d.doc_type,
            source_url=d.source_url,  # type: ignore[arg-type]
            retrieved_at=datetime.now(UTC),
            sha256=sha,
            raw_path=str(raw_rel),
            text=text,
        )
        manifest.append(data_dir, doc)
        landed.append(doc)
        seen.add(sha)
        log.info("doc_landed", id=doc.id, url=d.source_url)
    return landed


async def _default_fetch(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def _raw_path(city: str, d: DiscoveredDoc, sha: str) -> Path:
    date_part = d.meeting_date or "undated"
    safe_title = "".join(c if c.isalnum() else "-" for c in d.title)[:80].strip("-")
    suffix = Path(d.source_url.split("?", 1)[0]).suffix.lower() or ".bin"
    return Path("raw") / city / date_part / f"{safe_title}-{sha[:8]}{suffix}"


def _extract_text(path: Path) -> str:
    """Extract text from a downloaded file. Currently PDF-only via pypdf."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8", errors="ignore")
    from civic_slm.ingest.pdf import extract_pdf

    pages = extract_pdf(path)
    return "\n\n".join(p.text for p in pages if p.text)
