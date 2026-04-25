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

import os
import re
from dataclasses import dataclass
from pathlib import Path

# --- PII scrubbing ------------------------------------------------------------
#
# Public-meeting recordings routinely include residents stating their full
# name and home address during public-comment periods. Those residents
# expected a local audience, not a globally-distributed LLM training corpus.
# Default behavior:
#
#   * Speaker labels are replaced with `[Speaker]`.
#   * Lines inside a public-comment block (anything between a `>> Public
#     Comment` header and the next `>>` header that isn't another public-
#     comment one) are address-redacted.
#   * Address-shaped substrings anywhere are replaced with `[ADDRESS]`.
#
# Setting `CIVIC_SLM_KEEP_SPEAKER_NAMES=1` retains speaker labels — useful
# only for recipes whose speakers are unambiguously public figures (named
# elected officials, staff). The opt-out is documented in
# `docs/RECIPES.md` and `DATA_CARD.md`.

_KEEP_SPEAKERS_TRUTHY = {"1", "true", "yes", "on"}


def _keep_speaker_names() -> bool:
    raw = os.environ.get("CIVIC_SLM_KEEP_SPEAKER_NAMES", "").strip().lower()
    return raw in _KEEP_SPEAKERS_TRUTHY


_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[NSEW]\.?\s+)?[A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+){0,3}"
    r"\s+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Boulevard|Blvd\.?|Way|Court|Ct\.?|Place|Pl\.?|Circle|Cir\.?|Terrace|Ter\.?|Highway|Hwy\.?)\b",
    re.IGNORECASE,
)
_PUBLIC_COMMENT_HEADER_RE = re.compile(r"public\s+comment", re.IGNORECASE)


def _scrub_addresses(text: str) -> str:
    return _ADDRESS_RE.sub("[ADDRESS]", text)


def _is_public_comment_header(text: str) -> bool:
    return bool(_PUBLIC_COMMENT_HEADER_RE.search(text))


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


_SECTION_HEADER_RE = re.compile(r"^>{2,}|^Item\s+\d+", re.IGNORECASE)


def _public_comment_mask(lines: list[TranscriptLine]) -> list[bool]:
    """Per-line boolean: is this line inside a public-comment block?

    Treat the header line itself as outside the block so the speaker who
    *announces* the public-comment period (typically the mayor or chair)
    keeps their name. Subsequent lines are inside until a non-public-comment
    section header arrives.
    """
    mask: list[bool] = []
    in_public_comment = False
    for line in lines:
        if _is_public_comment_header(line.text):
            mask.append(False)
            in_public_comment = True
            continue
        if _SECTION_HEADER_RE.match(line.text):
            in_public_comment = False
        mask.append(in_public_comment)
    return mask


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
    """Group same-speaker runs into paragraphs separated by blank lines.

    Speakers drive paragraph breaks even when the rendered output is
    redacted, so a Mayor→commenter switch still creates a new paragraph.
    Speaker labels and addresses are scrubbed at render time per the PII
    policy in `DATA_CARD.md`.
    """
    if not lines:
        return ""
    keep_names = _keep_speaker_names()
    public_comment = _public_comment_mask(lines)
    paragraphs: list[str] = []
    current_speaker = lines[0].speaker
    current_in_pc = public_comment[0] if public_comment else False
    buf: list[str] = []
    for line, in_pc in zip(lines, public_comment, strict=False):
        if line.speaker != current_speaker and buf:
            paragraphs.append(
                _render_paragraph(buf, current_speaker, current_in_pc, keep_names=keep_names)
            )
            buf = []
            current_speaker = line.speaker
            current_in_pc = in_pc
        buf.append(line.text)
        # Cap paragraphs at ~30 lines so the chunker has somewhere to split.
        if len(buf) >= 30:
            paragraphs.append(
                _render_paragraph(buf, current_speaker, current_in_pc, keep_names=keep_names)
            )
            buf = []
    if buf:
        paragraphs.append(
            _render_paragraph(buf, current_speaker, current_in_pc, keep_names=keep_names)
        )
    return "\n\n".join(paragraphs)


def _render_paragraph(
    lines: list[str],
    speaker: str | None,
    in_public_comment: bool,
    *,
    keep_names: bool,
) -> str:
    body = _scrub_addresses(" ".join(lines))
    if speaker is None:
        return body
    display = speaker if (keep_names and not in_public_comment) else "[Speaker]"
    return f"{display}: {body}"
