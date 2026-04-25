"""yt-dlp wrapper for video discovery + audio/caption fetch.

Two surfaces:

  - `list_channel_videos(channel_url, since, max_videos)` enumerates a
    channel or playlist via `yt-dlp --flat-playlist --print-json`. We get
    one JSON line per video with id, title, upload_date, and duration —
    enough to populate `DiscoveredVideo` without downloading any media.
  - `fetch_audio_and_captions(video_url, out_dir)` downloads `bestaudio`
    as `.m4a` plus any human-uploaded subtitles and the auto-caption
    track. Returns paths so the orchestrator can pick the best source.

Why subprocess yt-dlp and not the Python API: yt-dlp's CLI is its
public API. The Python entry points exist but aren't stable across
releases. Shelling out is ugly but doesn't break when yt-dlp ships a
breaking change in `YoutubeDL.params`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

YT_DLP = "yt-dlp"


@dataclass(frozen=True)
class YouTubeVideoMeta:
    id: str
    title: str
    upload_date: str | None  # YYYYMMDD per yt-dlp
    duration_s: float | None
    webpage_url: str


@dataclass(frozen=True)
class FetchedMedia:
    """Paths returned by `fetch_audio_and_captions`. Any field may be None."""

    audio_path: Path | None
    human_subs: Path | None
    auto_subs: Path | None


def _yt_dlp_or_raise() -> str:
    if shutil.which(YT_DLP) is None:
        raise RuntimeError(
            "yt-dlp not found on PATH. Install it: `pip install yt-dlp` or "
            "`uv sync --extra ingest` (the ingest extra includes it)."
        )
    return YT_DLP


def list_channel_videos(
    channel_url: str,
    *,
    max_videos: int = 50,
    since: str | None = None,
) -> list[YouTubeVideoMeta]:
    """Enumerate videos in a channel/playlist. Doesn't download media.

    `since` is an ISO date (YYYY-MM-DD); we convert to yt-dlp's `--dateafter
    YYYYMMDD` format. Videos older than `since` are filtered server-side.
    """
    bin_ = _yt_dlp_or_raise()
    cmd = [
        bin_,
        "--flat-playlist",
        "--print-json",
        "--playlist-end",
        str(max_videos),
        "--no-warnings",
    ]
    if since:
        cmd.extend(["--dateafter", since.replace("-", "")])
    cmd.append(channel_url)

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out: list[YouTubeVideoMeta] = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(
            YouTubeVideoMeta(
                id=str(data.get("id", "")),
                title=str(data.get("title", "")),
                upload_date=str(data["upload_date"]) if data.get("upload_date") else None,
                duration_s=float(data["duration"]) if data.get("duration") else None,
                webpage_url=str(data.get("webpage_url") or data.get("url") or ""),
            )
        )
    return out


def fetch_audio_and_captions(
    video_url: str,
    *,
    out_dir: Path,
    sub_langs: tuple[str, ...] = ("en", "en-US", "en-GB"),
) -> FetchedMedia:
    """Download `.m4a` audio + (human + auto) captions for a single video.

    Returned paths point into `out_dir`. Missing artifacts (no audio, no
    captions) come back as None — the caller picks the best available.
    """
    bin_ = _yt_dlp_or_raise()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    sub_lang_arg = ",".join(sub_langs)
    cmd = [
        bin_,
        "-f",
        "bestaudio",
        "-x",
        "--audio-format",
        "m4a",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        sub_lang_arg,
        "--sub-format",
        "vtt",
        "--convert-subs",
        "vtt",
        "--no-playlist",
        "--no-warnings",
        "-o",
        out_template,
        video_url,
    ]
    subprocess.run(cmd, check=True)

    # yt-dlp writes <id>.<ext>. Find them.
    # Audio: <id>.m4a
    # Captions: <id>.<lang>.vtt for human, <id>.<lang>.vtt for auto-generated
    # (yt-dlp distinguishes via filename pattern; we resolve below).
    return _resolve_artifacts(out_dir, video_url)


def _resolve_artifacts(out_dir: Path, video_url: str) -> FetchedMedia:
    """yt-dlp doesn't return paths directly; resolve them by globbing."""
    # Extract video id from URL — naive but works for canonical YouTube URLs.
    video_id = video_url.rsplit("v=", 1)[-1].split("&", 1)[0] if "v=" in video_url else None
    audio: Path | None = None
    human: Path | None = None
    auto: Path | None = None
    pattern_id = video_id or "*"
    for path in out_dir.glob(f"{pattern_id}*"):
        name = path.name
        suffix = path.suffix.lower()
        if suffix == ".m4a":
            audio = path
        elif suffix == ".vtt":
            # yt-dlp marks auto-captions with `.<lang>.vtt` and human subs the same;
            # they are distinguished by an `auto-` prefix in some yt-dlp versions
            # and by file content (`Kind: captions` for auto). Fall back to file
            # content sniff: auto-generated VTT files usually contain the literal
            # `Kind: captions` header.
            if "auto" in name.lower() or _looks_auto(path):
                auto = path
            else:
                human = path
    return FetchedMedia(audio_path=audio, human_subs=human, auto_subs=auto)


def _looks_auto(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512]
    except OSError:
        return False
    return "Kind: captions" in head or "Language:" in head
