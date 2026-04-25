"""ASR fallback for videos without usable captions. mlx-whisper on Apple Silicon.

This module is **lazily imported by `transcript.extract_transcript()`** —
the rest of the package never imports it, so users on Linux (or anyone
without mlx-whisper installed) don't pay the cost. If the dep is missing
when ASR is actually needed, we raise a clear error pointing at the
install command.

Default model: `mlx-community/whisper-large-v3-turbo`. ~1x real-time on
M-series, accuracy on par with `large-v3`. Override via
`CIVIC_SLM_WHISPER_MODEL` for a smaller/faster model on memory-tight Macs.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def whisper_model() -> str:
    return os.environ.get("CIVIC_SLM_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


def transcribe(audio_path: Path, *, model: str | None = None) -> str:
    """Run mlx-whisper on the audio file and return plain text.

    Raises `RuntimeError` if mlx-whisper isn't installed (Linux users, or
    anyone who skipped the `train` extra). The message names the install
    command so it's actionable.
    """
    try:
        import mlx_whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper not installed. Install with: "
            "`uv sync --extra ingest` (Apple Silicon only) or "
            "fall back to caption-only by ensuring videos have captions."
        ) from exc

    result = mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType]
        str(audio_path),
        path_or_hf_repo=model or whisper_model(),
    )
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str):
        raise RuntimeError(f"mlx-whisper returned unexpected result shape: {type(result)}")
    return text.strip()
