"""Parse VTT and SRT subtitle files into plain transcript text.

Why this lives here and not in pdf.py: subtitle formats need cue dedup and
speaker-tag handling that don't apply to PDFs. YouTube auto-captions roll
the same words across consecutive cues (each cue overlaps the previous one
by about a second to keep the on-screen text steady), so naively
concatenating cue lines produces 3-4x duplication. We dedupe by tracking
the previous cue's tail and only emitting the new tail.

Speaker handling is a heuristic, not a model. We preserve three patterns
when they appear, otherwise the transcript is flat text:
  - `>> Speaker Name:` (NCAA-style closed-caption convention)
  - `<v Speaker Name>...` (VTT speaker voice tag)
  - `Speaker Name: ...` (already-formatted line)

This is intentionally cheap. Real diarization is a v1 line item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# WEBVTT cue header line: `00:00:01.000 --> 00:00:04.000` with optional settings.
_VTT_CUE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*$")
# SRT cue header line: `00:00:01,000 --> 00:00:04,000`.
_SRT_CUE_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}.*$")
# VTT voice tag: `<v Speaker Name>...</v>` or `<v Speaker Name>...`.
_VTT_VOICE_RE = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>|$)", re.IGNORECASE | re.DOTALL)
# Inline VTT styling tags we strip: `<c>`, `<i>`, `<00:00:01.000>`, etc.
_VTT_STYLE_RE = re.compile(r"<[^>]+>")
# `>> Speaker:` or `>>> Speaker:` close-caption pattern.
_DOUBLE_ANGLE_RE = re.compile(r"^>{2,}\s*([^:]+?):\s*(.*)$")


@dataclass(frozen=True)
class TranscriptLine:
    text: str
    speaker: str | None = None

    def render(self) -> str:
        return f"{self.speaker}: {self.text}" if self.speaker else self.text


def parse_vtt(path: Path) -> str:
    """Return cue-deduped plain text. VTT and Whisper-VTT both work here."""
    return _render(_iter_vtt_cues(path.read_text(encoding="utf-8", errors="ignore")))


def parse_srt(path: Path) -> str:
    return _render(_iter_srt_cues(path.read_text(encoding="utf-8", errors="ignore")))


def parse_subtitle(path: Path) -> str:
    """Pick the right parser by extension. Falls back to VTT for unknown."""
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt(path)
    return parse_vtt(path)


# --- internals ----------------------------------------------------------------


def _iter_vtt_cues(text: str) -> list[str]:
    """Walk a VTT body, emit cue text blocks (one per cue, may have multi-line content)."""
    lines = text.splitlines()
    cues: list[str] = []
    i = 0
    while i < len(lines):
        if _VTT_CUE_RE.match(lines[i]):
            i += 1
            buf: list[str] = []
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i])
                i += 1
            cues.append("\n".join(buf))
        else:
            i += 1
    return cues


def _iter_srt_cues(text: str) -> list[str]:
    lines = text.splitlines()
    cues: list[str] = []
    i = 0
    while i < len(lines):
        if _SRT_CUE_RE.match(lines[i]):
            i += 1
            buf: list[str] = []
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i])
                i += 1
            cues.append("\n".join(buf))
        else:
            i += 1
    return cues


def _render(cues: list[str]) -> str:
    """Cue blocks → newline-joined transcript with rolling dedup + speaker preservation.

    YouTube auto-captions emit each phrase across several cues, growing the
    text by one or two words per cue ("Welcome" → "Welcome everyone" →
    "Welcome everyone to the meeting"). We collapse those into a single
    line by treating each new cue as *extending* the previous when it
    starts with what we already have. Switching speakers or fully-disjoint
    text starts a new line.
    """
    out: list[TranscriptLine] = []
    for cue in cues:
        for line in _cue_lines(cue):
            if not line.text:
                continue
            if out and out[-1].speaker == line.speaker:
                prev = out[-1].text
                if line.text == prev:
                    continue
                if line.text.startswith(prev):
                    # Extension: replace the prior with the longer text.
                    out[-1] = TranscriptLine(text=line.text, speaker=line.speaker)
                    continue
                if prev.startswith(line.text):
                    # New cue is a strict prefix of what we've already got. Skip.
                    continue
            out.append(line)
    return _format_paragraphs(out)


def _cue_lines(cue: str) -> list[TranscriptLine]:
    """One cue can span multiple lines; emit a TranscriptLine per emitted line."""
    raw_lines = cue.split("\n")
    out: list[TranscriptLine] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        # `<v Speaker>...` voice tag.
        voice = _VTT_VOICE_RE.search(line)
        if voice:
            speaker = voice.group(1).strip()
            content = _VTT_STYLE_RE.sub("", line[voice.end(1) :]).strip().lstrip(">").strip()
            # Strip the matched voice block too.
            content = _VTT_VOICE_RE.sub(lambda m: m.group(2), content).strip()
            if content:
                out.append(TranscriptLine(text=content, speaker=speaker))
            continue
        # `>> Speaker:` pattern.
        m = _DOUBLE_ANGLE_RE.match(line)
        if m:
            out.append(TranscriptLine(text=m.group(2).strip(), speaker=m.group(1).strip()))
            continue
        # Plain `Speaker: text` (only when the prefix is short and looks like a name).
        if ":" in line:
            head, _, tail = line.partition(":")
            if 1 <= len(head.split()) <= 4 and head[:1].isupper() and tail.strip():
                out.append(TranscriptLine(text=tail.strip(), speaker=head.strip()))
                continue
        # Default: strip styling tags and emit flat.
        clean = _VTT_STYLE_RE.sub("", line).strip()
        if clean:
            out.append(TranscriptLine(text=clean))
    return out


def _format_paragraphs(lines: list[TranscriptLine]) -> str:
    """Group same-speaker runs into paragraphs separated by blank lines."""
    if not lines:
        return ""
    paragraphs: list[str] = []
    current_speaker = lines[0].speaker
    buf: list[str] = []
    for line in lines:
        if line.speaker != current_speaker and buf:
            paragraphs.append(_render_paragraph(buf, current_speaker))
            buf = []
            current_speaker = line.speaker
        buf.append(line.text)
        # Cap paragraphs at ~30 lines so the chunker has somewhere to split.
        if len(buf) >= 30:
            paragraphs.append(_render_paragraph(buf, current_speaker))
            buf = []
    if buf:
        paragraphs.append(_render_paragraph(buf, current_speaker))
    return "\n\n".join(paragraphs)


def _render_paragraph(lines: list[str], speaker: str | None) -> str:
    body = " ".join(lines)
    return f"{speaker}: {body}" if speaker else body
