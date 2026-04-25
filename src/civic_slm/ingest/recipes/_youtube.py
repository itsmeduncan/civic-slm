"""Shared YouTube discovery helpers for recipes.

Why this exists alongside `_browser.py`: browser-use is the right tool for
hand-built government sites (Granicus, custom CMS), but for YouTube
channels and playlists, `yt-dlp` already understands the URL structure
and can enumerate everything via JSON without launching a browser. Use
this module from your recipe's `discover_videos` method.

Typical usage from a recipe:

    from civic_slm.ingest.recipes._youtube import youtube_channel_videos

    async def discover_videos(self, *, since, max_videos):
        return youtube_channel_videos(
            "https://www.youtube.com/@san-clemente-tv/videos",
            since=since,
            max_videos=max_videos,
        )
"""

from __future__ import annotations

from civic_slm.ingest.harness import DiscoveredVideo
from civic_slm.ingest.video.youtube import list_channel_videos


def _meta_to_discovered(meta_list: list[object]) -> list[DiscoveredVideo]:
    out: list[DiscoveredVideo] = []
    for m in meta_list:
        upload_date = getattr(m, "upload_date", None)
        meeting_date: str | None = None
        if isinstance(upload_date, str) and len(upload_date) == 8:
            meeting_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        out.append(
            DiscoveredVideo(
                title=getattr(m, "title", ""),
                video_url=getattr(m, "webpage_url", ""),
                meeting_date=meeting_date,
                duration_s=getattr(m, "duration_s", None),
            )
        )
    return out


def youtube_channel_videos(
    channel_url: str, *, since: str | None = None, max_videos: int = 50
) -> list[DiscoveredVideo]:
    """Enumerate a channel or `/videos` page; map to `DiscoveredVideo`."""
    metas = list_channel_videos(channel_url, max_videos=max_videos, since=since)
    return _meta_to_discovered(list(metas))


def youtube_playlist_videos(playlist_url: str, *, max_videos: int = 50) -> list[DiscoveredVideo]:
    """Enumerate a playlist. Same plumbing — yt-dlp doesn't distinguish."""
    metas = list_channel_videos(playlist_url, max_videos=max_videos, since=None)
    return _meta_to_discovered(list(metas))
