"""Caption-first transcript extraction.

Single entrypoint: `extract_transcript(media)` takes a `FetchedMedia`
(from yt-dlp) and walks the priority chain:

  1. Human-uploaded SRT/VTT  → `transcript_source = "human_srt"` or `"vtt"`
  2. YouTube auto-caption    → `transcript_source = "youtube_caption"`
  3. mlx-whisper ASR on audio → `transcript_source = "whisper"`

Falling all the way through (no captions, no audio) raises so the caller
knows the manifest entry should be skipped, not silently empty.
"""

from __future__ import annotations

from civic_slm.ingest.video.caption import parse_subtitle
from civic_slm.ingest.video.youtube import FetchedMedia
from civic_slm.schema import TranscriptSource


def extract_transcript(media: FetchedMedia) -> tuple[str, TranscriptSource]:
    """Walk the priority chain. Returns (text, source) on first success."""
    if media.human_subs and media.human_subs.exists():
        text = parse_subtitle(media.human_subs)
        if text:
            source: TranscriptSource = "vtt" if media.human_subs.suffix == ".vtt" else "human_srt"
            return text, source
    if media.auto_subs and media.auto_subs.exists():
        text = parse_subtitle(media.auto_subs)
        if text:
            return text, "youtube_caption"
    if media.audio_path and media.audio_path.exists():
        # Lazy import keeps the module loadable on Linux / without mlx-whisper.
        from civic_slm.ingest.video.asr import transcribe

        text = transcribe(media.audio_path)
        if text:
            return text, "whisper"
    raise RuntimeError(
        "No usable transcript source: no captions and no audio. Caller should skip this video."
    )
