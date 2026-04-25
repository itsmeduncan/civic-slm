"""VTT/SRT parser tests — fixtures inline so no external files needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from civic_slm.ingest.video.caption import parse_subtitle


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_vtt_basic_parses_to_text(tmp_path: Path) -> None:
    body = """WEBVTT

00:00:01.000 --> 00:00:04.000
Welcome to the City Council meeting.

00:00:04.500 --> 00:00:08.000
The first item on the agenda is the budget.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    assert "Welcome to the City Council meeting." in out
    assert "first item on the agenda" in out


def test_vtt_dedupes_rolling_youtube_captions(tmp_path: Path) -> None:
    """YouTube auto-captions repeat each cue's prefix in the next cue."""
    body = """WEBVTT

00:00:00.000 --> 00:00:02.000
Welcome everyone

00:00:02.000 --> 00:00:04.000
Welcome everyone to the meeting

00:00:04.000 --> 00:00:06.000
Welcome everyone to the meeting tonight
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    # The full final phrase should appear, but not three near-duplicate copies.
    assert "Welcome everyone" in out
    assert out.count("Welcome everyone") == 1


def test_vtt_voice_tag_scrubs_speaker_by_default(tmp_path: Path) -> None:
    body = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Mayor Whitfield>I call this meeting to order.

00:00:05.000 --> 00:00:08.000
<v Councilmember Ramirez>Thank you, Mayor.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    assert "Mayor Whitfield" not in out
    assert "Councilmember Ramirez" not in out
    assert "[Speaker]: I call this meeting to order." in out


def test_vtt_voice_tag_kept_when_opt_out_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIVIC_SLM_KEEP_SPEAKER_NAMES", "1")
    body = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Mayor Whitfield>I call this meeting to order.

00:00:05.000 --> 00:00:08.000
<v Councilmember Ramirez>Thank you, Mayor.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    assert "Mayor Whitfield: I call this meeting to order." in out
    assert "Councilmember Ramirez: Thank you, Mayor." in out


def test_public_comment_block_always_scrubs_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIVIC_SLM_KEEP_SPEAKER_NAMES", "1")
    body = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Mayor Whitfield>We will now hear public comment.

00:00:05.000 --> 00:00:08.000
<v Jane Resident>I live at 123 Main Street and oppose the project.

00:00:09.000 --> 00:00:12.000
Item 5. Continued business.

00:00:13.000 --> 00:00:16.000
<v Mayor Whitfield>Moving on to item five.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    # Mayor's own name is preserved outside the public-comment block.
    assert "Mayor Whitfield: We will now hear public comment." in out
    # Resident name is scrubbed regardless of opt-out.
    assert "Jane Resident" not in out
    assert "[Speaker]:" in out
    # Address is redacted.
    assert "123 Main Street" not in out
    assert "[ADDRESS]" in out
    # Mayor's name comes back after the public-comment block ends.
    assert "Mayor Whitfield: Moving on to item five." in out


def test_vtt_double_angle_speaker_pattern_scrubbed(tmp_path: Path) -> None:
    body = """WEBVTT

00:00:01.000 --> 00:00:04.000
>> MAYOR: I call this meeting to order.

00:00:05.000 --> 00:00:08.000
>> RAMIREZ: Thank you.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    assert "MAYOR" not in out
    assert "RAMIREZ" not in out
    assert "[Speaker]: I call this meeting to order." in out
    assert "[Speaker]: Thank you." in out


def test_srt_basic_parses(tmp_path: Path) -> None:
    body = """1
00:00:01,000 --> 00:00:04,000
Welcome to the City Council meeting.

2
00:00:04,500 --> 00:00:08,000
The first item is the budget.
"""
    out = parse_subtitle(_write(tmp_path, "x.srt", body))
    assert "Welcome to the City Council meeting." in out


@pytest.mark.parametrize("ext", ["vtt", "srt"])
def test_empty_file_returns_empty(tmp_path: Path, ext: str) -> None:
    out = parse_subtitle(_write(tmp_path, f"x.{ext}", ""))
    assert out == ""


def test_speaker_change_creates_paragraph_break(tmp_path: Path) -> None:
    body = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Mayor>First point.

00:00:03.000 --> 00:00:05.000
<v Mayor>Second point from the mayor.

00:00:05.000 --> 00:00:07.000
<v Ramirez>I disagree.
"""
    out = parse_subtitle(_write(tmp_path, "x.vtt", body))
    # Two paragraphs, separated by blank line.
    assert "\n\n" in out
    assert out.count("\n\n") == 1
