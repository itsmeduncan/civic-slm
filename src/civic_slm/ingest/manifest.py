"""Append-only manifest of crawled documents.

Why append-only: the manifest is the single source of truth that says "we
fetched this URL at this time and got these bytes." Treating it as a journal
makes re-runs idempotent (existing entries by sha256 are skipped) and keeps
the audit trail intact even if a city later changes the URL or pulls the doc.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from civic_slm.schema import CivicDocument

if TYPE_CHECKING:
    from collections.abc import Iterator


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "raw" / "manifest.jsonl"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(data_dir: Path) -> list[CivicDocument]:
    p = manifest_path(data_dir)
    if not p.exists():
        return []
    return [CivicDocument.model_validate_json(line) for line in _iter_lines(p)]


def known_hashes(data_dir: Path) -> set[str]:
    return {d.sha256 for d in load_manifest(data_dir)}


def append(data_dir: Path, doc: CivicDocument) -> None:
    p = manifest_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(doc.model_dump_json() + "\n")


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped
