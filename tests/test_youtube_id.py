"""URL → 11-char YouTube video ID parsing."""

from __future__ import annotations

import pytest

from civic_slm.ingest.video.youtube import (
    _extract_youtube_id,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_youtube_id_supported_shapes(url: str, expected: str) -> None:
    assert _extract_youtube_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        # Adversarial: SLD-ends-with-but-isn't `youtu.be`. The bug fix ensures
        # `endswith("youtu.be")` doesn't match `evilyoutu.be`.
        "https://evilyoutu.be/SOMETHING",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/feed/subscriptions",
        "not a url at all",
    ],
)
def test_extract_youtube_id_rejects_non_youtube_or_pathless(url: str) -> None:
    assert _extract_youtube_id(url) is None
