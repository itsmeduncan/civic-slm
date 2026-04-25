"""browser-use / browser-harness driver.

Each jurisdiction (city, county, township) has a *recipe* — a high-level
natural-language instruction plus a list of expected fields. The recipe is
handed to a browser-use Agent which navigates the jurisdiction's site and
returns a structured list of document URLs. The harness then downloads each
URL via httpx and writes it to data/raw/.

Why an agentic crawler instead of a hand-written scraper: U.S. local-government
sites run on a long tail of vendor platforms (Granicus, Legistar, CivicPlus,
IQM, PrimeGov, Municode, custom WordPress). An agent given "find council
agendas for the last 12 months" usually beats per-platform CSS selectors that
rot, and the same recipe pattern works whether the jurisdiction is a CA city,
a TX county, or a NY township. We pay for it in API tokens and wallclock;
at the v0 corpus scale (~thousands of docs, not millions) the tradeoff is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from civic_slm.ingest import manifest
from civic_slm.logging import get_logger
from civic_slm.schema import CivicDocument, DocType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
else:
    from collections.abc import Callable  # runtime: needed for crawl_videos signature

log = get_logger(__name__)


@dataclass(frozen=True)
class DiscoveredDoc:
    """One document the agent found, before we've fetched the bytes."""

    title: str
    source_url: str
    doc_type: DocType
    meeting_date: str | None = None  # ISO date string when relevant


@dataclass(frozen=True)
class DiscoveredVideo:
    """A meeting recording the recipe wants ingested + transcribed.

    `video_url` is a YouTube watch URL or a direct MP4. `meeting_date` is
    optional (some channels post videos without dates). `duration_s` lets
    the orchestrator skip oversize videos before download if you want.
    """

    title: str
    video_url: str
    meeting_date: str | None = None
    duration_s: float | None = None


class Recipe(Protocol):
    """A jurisdiction recipe — must expose `jurisdiction` + `state` and an async `discover`.

    `jurisdiction` is a kebab-case slug (e.g. `san-clemente`, `harris-county`,
    `new-york`). `state` is the 2-letter postal code. Together they uniquely
    locate a recipe across the U.S.

    Recipes that publish meeting recordings can also implement
    `discover_videos()`. The base orchestrator (`crawl_videos`) does nothing
    on recipes that omit it.
    """

    @property
    def jurisdiction(self) -> str: ...

    @property
    def state(self) -> str: ...

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
    log.info(
        "discovered_docs",
        jurisdiction=recipe.jurisdiction,
        state=recipe.state,
        count=len(discovered),
    )

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

        raw_rel = _raw_path(recipe.state, recipe.jurisdiction, d, sha)
        raw_abs = data_dir / raw_rel
        raw_abs.parent.mkdir(parents=True, exist_ok=True)
        raw_abs.write_bytes(data)

        text = _extract_text(raw_abs)
        if not text:
            log.warning("empty_extraction", url=d.source_url)
            continue

        doc = CivicDocument(
            id=f"{recipe.state}/{recipe.jurisdiction}/{sha[:12]}",
            jurisdiction=recipe.jurisdiction,
            state=recipe.state,
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


def _raw_path(state: str, jurisdiction: str, d: DiscoveredDoc, sha: str) -> Path:
    date_part = d.meeting_date or "undated"
    safe_title = "".join(c if c.isalnum() else "-" for c in d.title)[:80].strip("-")
    suffix = Path(d.source_url.split("?", 1)[0]).suffix.lower() or ".bin"
    return (
        Path("raw") / state.lower() / jurisdiction / date_part / f"{safe_title}-{sha[:8]}{suffix}"
    )


# --- video orchestration ------------------------------------------------------


def _video_dir(state: str, jurisdiction: str, v: DiscoveredVideo) -> Path:
    """Directory for the audio + caption + transcript artifacts of one video."""
    date_part = v.meeting_date or "undated"
    return Path("raw") / state.lower() / jurisdiction / date_part / "video"


async def crawl_videos(
    *,
    recipe: Any,  # type: ignore[valid-type]  # Recipe + optional discover_videos
    data_dir: Path,
    since: str,
    max_videos: int,
    fetch_media: Callable[[str, Path], object] | None = None,
    extract: Callable[[object], tuple[str, str]] | None = None,
) -> list[CivicDocument]:
    """Discover videos via the recipe, fetch audio + captions, transcribe, append.

    `fetch_media(video_url, out_dir) -> FetchedMedia` and
    `extract(media) -> (text, transcript_source)` are injectable for tests
    so the live yt-dlp / mlx-whisper paths don't run in CI.
    """
    if not hasattr(recipe, "discover_videos"):
        log.info("no_videos_for_recipe", jurisdiction=getattr(recipe, "jurisdiction", "?"))
        return []

    if fetch_media is None:
        from civic_slm.ingest.video.youtube import fetch_audio_and_captions

        def _fetch(url: str, out_dir: Path) -> object:
            return fetch_audio_and_captions(url, out_dir=out_dir)

        fetch_media = _fetch

    if extract is None:
        from civic_slm.ingest.video.transcript import extract_transcript

        def _extract(media: object) -> tuple[str, str]:
            text, source = extract_transcript(media)  # type: ignore[arg-type]
            return text, str(source)

        extract = _extract

    discovered = await recipe.discover_videos(since=since, max_videos=max_videos)
    log.info(
        "discovered_videos",
        jurisdiction=recipe.jurisdiction,
        state=recipe.state,
        count=len(discovered),
    )

    seen = manifest.known_hashes(data_dir)
    landed: list[CivicDocument] = []
    for v in discovered:
        # Dedup key: sha256 of the canonical video_url. We can't sha the audio
        # without downloading first (which is what we're trying to dedup).
        sha = manifest.sha256_bytes(v.video_url.encode("utf-8"))
        if sha in seen:
            continue

        rel_dir = _video_dir(recipe.state, recipe.jurisdiction, v)
        abs_dir = data_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        try:
            media = fetch_media(v.video_url, abs_dir)
        except Exception as exc:
            log.warning("fetch_media_failed", url=v.video_url, error=str(exc))
            continue

        try:
            text, source = extract(media)
        except Exception as exc:
            log.warning("transcript_failed", url=v.video_url, error=str(exc))
            continue

        if not text:
            log.warning("empty_transcript", url=v.video_url)
            continue

        audio_path = getattr(media, "audio_path", None)
        raw_path_str = str(audio_path.relative_to(data_dir)) if audio_path else str(rel_dir)

        doc = CivicDocument(
            id=f"{recipe.state}/{recipe.jurisdiction}/video/{sha[:12]}",
            jurisdiction=recipe.jurisdiction,
            state=recipe.state,
            doc_type=DocType.MEETING_TRANSCRIPT,
            source_url=v.video_url,  # type: ignore[arg-type]
            retrieved_at=datetime.now(UTC),
            sha256=sha,
            raw_path=raw_path_str,
            text=text,
            video_url=v.video_url,  # type: ignore[arg-type]
            transcript_source=source,  # type: ignore[arg-type]
            duration_s=v.duration_s,
        )
        manifest.append(data_dir, doc)
        landed.append(doc)
        seen.add(sha)
        log.info("video_landed", id=doc.id, source=source, url=v.video_url)
    return landed


def _extract_text(path: Path) -> str:
    """Extract text from a downloaded file. Currently PDF-only via pypdf."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8", errors="ignore")
    from civic_slm.ingest.pdf import extract_pdf

    pages = extract_pdf(path)
    return "\n\n".join(p.text for p in pages if p.text)
