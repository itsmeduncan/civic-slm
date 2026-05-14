"""Pin the san-clemente recipe's `discover_videos` contract.

#45 added a YouTube discovery path to the reference recipe so
`civic-slm crawl-videos san-clemente` no longer 404s. The test asserts
the method exists, returns the right shape, and that the YouTube
channel URL the recipe ships is the one quoted in `docs/SOURCES.md`
(those two locations have to stay in sync — auditors check SOURCES,
crawlers honor the recipe).

`yt-dlp` is mocked so the test stays offline and fast.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from civic_slm.ingest.harness import DiscoveredVideo
from civic_slm.ingest.recipes.san_clemente import (
    YOUTUBE_CHANNEL,
    SanClementeRecipe,
)

_SOURCES_PATH = Path("docs/SOURCES.md")


def _stub_yt_dlp(monkeypatch: pytest.MonkeyPatch, n: int = 3) -> None:
    """Install a fake `civic_slm.ingest.video.youtube.list_channel_videos`
    that returns N video metadata records without hitting YouTube.

    We mock at the project boundary (`list_channel_videos`) rather than
    yt-dlp itself because that's the project-controlled seam — yt-dlp's
    `YoutubeDL` API surface is large and changes between releases.
    """

    class _VideoMeta:
        def __init__(self, i: int) -> None:
            self.title = f"City Council Meeting {i}"
            self.webpage_url = f"https://www.youtube.com/watch?v=stub{i}"
            self.upload_date = f"2025010{i + 1}"  # YYYYMMDD
            self.duration_s = 60.0 * (i + 1)

    fake_module = types.ModuleType("civic_slm.ingest.video.youtube_stub")

    def list_channel_videos(
        channel_url: str, *, max_videos: int = 50, since: str | None = None
    ) -> list[_VideoMeta]:
        return [_VideoMeta(i) for i in range(min(n, max_videos))]

    fake_module.list_channel_videos = list_channel_videos  # type: ignore[attr-defined]
    # Patch the symbol the recipe's helper imports — `_youtube.list_channel_videos`.
    monkeypatch.setattr(
        "civic_slm.ingest.recipes._youtube.list_channel_videos",
        list_channel_videos,
        raising=True,
    )
    # Keep the fake module reachable in sys.modules for any indirect import.
    monkeypatch.setitem(sys.modules, "civic_slm.ingest.video.youtube_stub", fake_module)


def test_recipe_exposes_youtube_channel() -> None:
    recipe = SanClementeRecipe()
    assert recipe.youtube_channel == YOUTUBE_CHANNEL
    assert YOUTUBE_CHANNEL.startswith("https://www.youtube.com/")


def test_sources_md_references_recipe_channel_handle() -> None:
    """The ToS audit in SOURCES.md must cite the same channel the recipe
    crawls. A divergence here means the audit is for a different channel
    than what the crawler actually hits."""
    text = _SOURCES_PATH.read_text(encoding="utf-8")
    # Compare on the handle portion ("@san-clemente-tv") rather than the
    # exact URL — SOURCES.md sometimes lists the bare channel without `/videos`.
    handle_start = YOUTUBE_CHANNEL.find("/@")
    assert handle_start != -1
    handle_end = YOUTUBE_CHANNEL.find("/", handle_start + 1)
    handle = YOUTUBE_CHANNEL[handle_start + 1 : handle_end]
    assert handle in text, (
        f"SOURCES.md doesn't reference {handle!r}; the san-clemente audit "
        f"either doesn't cover the channel the recipe crawls, or the channel "
        f"URL in the recipe drifted from what was audited."
    )


def test_discover_videos_returns_discovered_video_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recipe must return `DiscoveredVideo` instances in the shape
    the harness expects, with `meeting_date` formatted as YYYY-MM-DD."""
    _stub_yt_dlp(monkeypatch, n=3)
    recipe = SanClementeRecipe()
    results = asyncio.run(recipe.discover_videos(since="2025-01-01", max_videos=10))
    assert len(results) == 3
    for r in results:
        assert isinstance(r, DiscoveredVideo)
        assert r.video_url.startswith("https://www.youtube.com/")
        # _youtube._meta_to_discovered() formats YYYYMMDD → YYYY-MM-DD.
        assert r.meeting_date and r.meeting_date.startswith("2025-01-")


def test_discover_videos_honors_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """`max_videos` flows through to the helper; the recipe doesn't drop it."""
    _stub_yt_dlp(monkeypatch, n=10)
    recipe = SanClementeRecipe()
    results = asyncio.run(recipe.discover_videos(since="2025-01-01", max_videos=2))
    assert len(results) == 2
