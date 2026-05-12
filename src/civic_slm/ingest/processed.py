from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from civic_slm.config import settings
from civic_slm.schema import DocumentChunk


def processed_path(jurisdiction: str, data_dir: Path | None = None) -> Path:
    """Resolve the processed-chunks JSONL path for a jurisdiction."""
    base = data_dir or settings().data_dir
    return base / "processed" / f"{jurisdiction}.jsonl"


def save_chunks(
    jurisdiction: str,
    chunks: Iterable[DocumentChunk],
    data_dir: Path | None = None,
) -> Path:
    """Write `DocumentChunk`s to data/processed/{jurisdiction}.jsonl.

    Overwrites any prior file for this jurisdiction so the processed view stays
    a function of the current manifest.
    """
    out = processed_path(jurisdiction, data_dir=data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    return out


def load_chunks(jurisdiction: str, data_dir: Path | None = None) -> list[DocumentChunk]:
    """Read processed chunks back from disk."""
    p = processed_path(jurisdiction, data_dir=data_dir)
    if not p.exists():
        return []
    out: list[DocumentChunk] = []
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            out.append(DocumentChunk.model_validate_json(line))
    return out
