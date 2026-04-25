"""PDF text extraction and section-aware chunking.

Heuristic chunker: walk pages, treat ALL-CAPS or numbered headings as section
boundaries, accumulate paragraphs into chunks of ~1024 tokens with 128-token
overlap. Token count is approximated as `len(text.split())` — close enough for
batch sizing decisions; real tokenization happens at training time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from civic_slm.schema import DocumentChunk

TARGET_TOKENS = 1024
OVERLAP_TOKENS = 128

_HEADING_RE = re.compile(
    r"""^
        (
            [A-Z][A-Z0-9 \-/&]{3,}      # ALL CAPS heading
            |
            \d+(\.\d+)*\.?\s+[A-Z].*    # numbered: "3.1 Land Use"
            |
            (Section|Article|Chapter|Item)\s+\d+.*
        )$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ExtractedPage:
    page_idx: int
    text: str


def extract_pdf(path: Path) -> list[ExtractedPage]:
    """Extract text from a PDF, one entry per page."""
    from pypdf import PdfReader  # type: ignore[import-not-found]  # optional dep, lazy import

    reader = PdfReader(str(path))
    return [
        ExtractedPage(page_idx=i, text=(page.extract_text() or "").strip())
        for i, page in enumerate(reader.pages)
    ]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    return _HEADING_RE.match(line) is not None


def _approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_text(
    doc_id: str,
    text: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    source_doc_hash: str | None = None,
) -> list[DocumentChunk]:
    """Split a document's text into overlapping chunks with section context.

    `source_doc_hash` should be the upstream `CivicDocument.sha256`; it is
    propagated onto every chunk so synth and the eval contamination check can
    bind back to the source.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return list(_pack(doc_id, paragraphs, target_tokens, overlap_tokens, source_doc_hash))


def _pack(
    doc_id: str,
    paragraphs: list[str],
    target: int,
    overlap: int,
    source_doc_hash: str | None,
) -> Iterator[DocumentChunk]:
    section: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    chunk_idx = 0

    for para in paragraphs:
        if _is_heading(para):
            section = [*section[:2], para] if len(section) >= 3 else [*section, para]
            continue

        para_tokens = _approx_tokens(para)
        if buf_tokens + para_tokens > target and buf:
            chunk_text_str = "\n\n".join(buf)
            yield DocumentChunk(
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                text=chunk_text_str,
                token_count=buf_tokens,
                section_path=section.copy(),
                source_doc_hash=source_doc_hash,
            )
            chunk_idx += 1
            buf, buf_tokens = _tail_overlap(buf, overlap)
        buf.append(para)
        buf_tokens += para_tokens

    if buf:
        yield DocumentChunk(
            doc_id=doc_id,
            chunk_idx=chunk_idx,
            text="\n\n".join(buf),
            token_count=buf_tokens,
            section_path=section.copy(),
            source_doc_hash=source_doc_hash,
        )


def _tail_overlap(paragraphs: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the trailing paragraphs whose combined token count ≈ overlap_tokens."""
    tail: list[str] = []
    total = 0
    for para in reversed(paragraphs):
        tokens = _approx_tokens(para)
        if total + tokens > overlap_tokens and tail:
            break
        tail.insert(0, para)
        total += tokens
    return tail, total
