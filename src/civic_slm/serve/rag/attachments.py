"""Turn an uploaded civic document into grounding text.

No FastAPI import here so the extraction logic is unit-testable in isolation;
the shim route in cli.py is a thin HTTP wrapper over `extract_document`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 30_000
_SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedDocType(ValueError):
    """Raised for a file extension outside `.pdf/.txt/.md`."""


class DocExtractionError(RuntimeError):
    """Raised when a supported file can't be read (encrypted/corrupt PDF)."""


@dataclass(frozen=True)
class DocText:
    filename: str
    content_type: str
    pages: int
    chars: int
    truncated: bool
    text: str


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CHARS:
        return text, False
    marker = "\n[document truncated]"
    return text[: MAX_CHARS - len(marker)] + marker, True


def extract_document(filename: str, data: bytes) -> DocText:
    ext = Path(filename).suffix.lower()
    if ext not in _SUPPORTED:
        raise UnsupportedDocType(f"unsupported type {ext!r}; allowed: .pdf, .txt, .md")

    if ext in {".txt", ".md"}:
        raw = data.decode("utf-8", errors="replace").strip()
        text, truncated = _truncate(raw)
        ctype = "text/markdown" if ext == ".md" else "text/plain"
        return DocText(filename, ctype, 1, len(text), truncated, text)

    # PDF: extract_pdf takes a path, so stage to a temp file.
    from civic_slm.ingest.pdf import extract_pdf  # lazy: pypdf is the `ingest` extra

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(data)
        fh.flush()
        try:
            pages = extract_pdf(Path(fh.name))
        except Exception as exc:  # pypdf raises PdfReadError etc.
            raise DocExtractionError(f"could not read PDF {filename!r}: {exc}") from exc
    raw = "\n\n".join(p.text for p in pages if p.text).strip()
    if not raw:
        raise DocExtractionError(f"no extractable text in {filename!r} (scanned image?)")
    text, truncated = _truncate(raw)
    return DocText(filename, "application/pdf", len(pages), len(text), truncated, text)
