"""crawl_videos orchestrator: stub yt-dlp + stub transcript extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from civic_slm.ingest import manifest
from civic_slm.ingest.harness import DiscoveredVideo, crawl_videos
from civic_slm.schema import DocType


@dataclass
class _FakeRecipe:
    jurisdiction: str = "test-city"
    state: str = "CA"
    videos: list[DiscoveredVideo] = ()  # type: ignore[assignment]

    async def discover_videos(self, *, since: str, max_videos: int) -> list[DiscoveredVideo]:
        return list(self.videos)


@dataclass
class _StubMedia:
    audio_path: Path | None
    human_subs: Path | None = None
    auto_subs: Path | None = None


async def test_crawl_videos_lands_transcript(tmp_path: Path) -> None:
    videos = [
        DiscoveredVideo(
            title="Council Meeting 2026-01-15",
            video_url="https://www.youtube.com/watch?v=abc123",
            meeting_date="2026-01-15",
            duration_s=3600.0,
        )
    ]
    recipe = _FakeRecipe(videos=videos)

    audio_path = tmp_path / "abc123.m4a"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"\x00\x00")

    def fake_fetch(url: str, out_dir: Path) -> object:
        return _StubMedia(audio_path=audio_path)

    def fake_extract(media: object) -> tuple[str, str]:
        return ("Mayor: Welcome.\n\nRamirez: Thank you.", "youtube_caption")

    landed = await crawl_videos(
        recipe=recipe,
        data_dir=tmp_path,
        since="2026-01-01",
        max_videos=10,
        fetch_media=fake_fetch,
        extract=fake_extract,
    )
    assert len(landed) == 1
    doc = landed[0]
    assert doc.doc_type == DocType.MEETING_TRANSCRIPT
    assert doc.transcript_source == "youtube_caption"
    assert "Mayor: Welcome." in doc.text
    rows = manifest.load_manifest(tmp_path)
    assert len(rows) == 1


async def test_crawl_videos_idempotent(tmp_path: Path) -> None:
    videos = [
        DiscoveredVideo(
            title="X",
            video_url="https://www.youtube.com/watch?v=zzz",
            meeting_date="2026-02-01",
        )
    ]
    recipe = _FakeRecipe(videos=videos)

    def fake_fetch(url: str, out_dir: Path) -> object:
        return _StubMedia(audio_path=None)

    def fake_extract(media: object) -> tuple[str, str]:
        return ("transcript", "vtt")

    first = await crawl_videos(
        recipe=recipe,
        data_dir=tmp_path,
        since="2026-01-01",
        max_videos=10,
        fetch_media=fake_fetch,
        extract=fake_extract,
    )
    assert len(first) == 1
    second = await crawl_videos(
        recipe=recipe,
        data_dir=tmp_path,
        since="2026-01-01",
        max_videos=10,
        fetch_media=fake_fetch,
        extract=fake_extract,
    )
    assert second == []


async def test_crawl_videos_skips_recipes_without_video_support(tmp_path: Path) -> None:
    @dataclass
    class _NoVideoRecipe:
        jurisdiction: str = "x"
        state: str = "CA"

    landed = await crawl_videos(
        recipe=_NoVideoRecipe(),
        data_dir=tmp_path,
        since="2026-01-01",
        max_videos=10,
        fetch_media=lambda url, d: _StubMedia(audio_path=None),
        extract=lambda m: ("t", "vtt"),
    )
    assert landed == []


async def test_crawl_videos_skips_on_empty_transcript(tmp_path: Path) -> None:
    videos = [
        DiscoveredVideo(
            title="Empty",
            video_url="https://www.youtube.com/watch?v=empty",
        )
    ]
    recipe = _FakeRecipe(videos=videos)

    def fake_extract(media: object) -> tuple[str, str]:
        return ("", "youtube_caption")

    landed = await crawl_videos(
        recipe=recipe,
        data_dir=tmp_path,
        since="2026-01-01",
        max_videos=10,
        fetch_media=lambda url, d: _StubMedia(audio_path=None),
        extract=fake_extract,
    )
    assert landed == []
    assert manifest.load_manifest(tmp_path) == []
