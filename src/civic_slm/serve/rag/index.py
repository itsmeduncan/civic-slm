"""Build an embedding index over a jurisdiction's processed chunks.

Phase 3 of the snazzy-tinkering-stardust plan, scope-expanded with the user's
approval (see `CLAUDE.md` "Out of scope" — local single-jurisdiction RAG for
dogfooding is in scope; multi-tenant production serving stays out).

Design: numpy + sentence-transformers is enough for a small civic corpus.
A 200-chunk corpus at 1024-dim float16 fits in ~400KB; cosine over that is
~microseconds. We don't need LanceDB / FAISS / pgvector for one
jurisdiction's meeting agendas, and pulling them in would add an extra
dependency tier for a problem that doesn't have a scale.

Output layout under `artifacts/<slug>-rag/`:

    embeddings.npy   chunks x dim, float16, normalized to unit length
    index.jsonl      one record per chunk with citation metadata, line-
                     aligned with `embeddings.npy` row order

The two files together are the index. They're regenerable from
`data/processed/<slug>.jsonl` + `data/raw/manifest.jsonl`, so we don't
need to checksum them or version-pin the embedding model — re-run if
either input changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from civic_slm.eval.embeddings import DEFAULT_BGE_MODEL
from civic_slm.ingest.manifest import load_manifest
from civic_slm.ingest.processed import load_chunks
from civic_slm.logging import get_logger
from civic_slm.schema import CivicDocument, DocumentChunk

log = get_logger(__name__)


@dataclass(frozen=True)
class IndexRecord:
    """One row in the index. Stays narrow on purpose: this lands on disk
    as JSONL and the columns are what we need to cite a passage in a
    user-facing answer."""

    chunk_id: str
    doc_id: str
    chunk_idx: int
    text: str
    source_url: str
    meeting_date: str | None
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_idx": self.chunk_idx,
            "text": self.text,
            "source_url": self.source_url,
            "meeting_date": self.meeting_date,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexRecord:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            chunk_idx=int(data["chunk_idx"]),
            text=data["text"],
            source_url=data["source_url"],
            meeting_date=data.get("meeting_date"),
            sha256=data["sha256"],
        )


def _join_chunks_to_manifest(
    chunks: list[DocumentChunk], manifest: list[CivicDocument]
) -> list[IndexRecord]:
    """Stitch each chunk to its parent `CivicDocument` via `source_doc_hash`.

    Chunks without a `source_doc_hash` (purely synthetic / orphaned) get a
    synthetic source row with empty citation fields; the answer layer can
    still cite by `chunk_id` and the user sees "no provenance metadata"
    rather than a crash.
    """
    by_hash = {d.sha256: d for d in manifest}
    records: list[IndexRecord] = []
    for c in chunks:
        parent = by_hash.get(c.source_doc_hash) if c.source_doc_hash else None
        chunk_id = f"{c.doc_id}#{c.chunk_idx}"
        if parent is None:
            records.append(
                IndexRecord(
                    chunk_id=chunk_id,
                    doc_id=c.doc_id,
                    chunk_idx=c.chunk_idx,
                    text=c.text,
                    source_url="",
                    meeting_date=None,
                    sha256=c.source_doc_hash or "",
                )
            )
            continue
        # `meeting_date` lives on `DiscoveredDoc`, not `CivicDocument` —
        # by the time we have a CivicDocument the meeting date has been
        # baked into the raw_path / id but isn't a first-class field. Use
        # the retrieval timestamp's date as the best available proxy.
        records.append(
            IndexRecord(
                chunk_id=chunk_id,
                doc_id=c.doc_id,
                chunk_idx=c.chunk_idx,
                text=c.text,
                source_url=str(parent.source_url),
                meeting_date=parent.retrieved_at.date().isoformat(),
                sha256=parent.sha256,
            )
        )
    return records


def build_index(
    jurisdiction: str,
    *,
    data_dir: Path,
    out_dir: Path,
    embedding_model: str = DEFAULT_BGE_MODEL,
) -> tuple[Path, Path]:
    """Build `embeddings.npy` + `index.jsonl` for one jurisdiction.

    Returns the two output paths so callers (CLI, tests) can assert
    presence without re-deriving them.
    """
    chunks = load_chunks(jurisdiction, data_dir=data_dir)
    if not chunks:
        raise ValueError(
            f"No processed chunks for {jurisdiction!r}. "
            f"Run `civic-slm process {jurisdiction}` first."
        )
    manifest = load_manifest(data_dir)
    records = _join_chunks_to_manifest(chunks, manifest)
    log.info("rag_index_build", jurisdiction=jurisdiction, chunks=len(records))

    # Lazy import — sentence-transformers is in the `eval` extra only.
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    encoder = SentenceTransformer(embedding_model)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    embeddings = encoder.encode(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        [r.text for r in records],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.asarray(embeddings, dtype=np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "embeddings.npy"
    jsonl_path = out_dir / "index.jsonl"
    np.save(npy_path, arr)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    log.info(
        "rag_index_written",
        jurisdiction=jurisdiction,
        embeddings=str(npy_path),
        index=str(jsonl_path),
        rows=arr.shape[0],
        dim=arr.shape[1] if arr.ndim == 2 else None,
    )
    return npy_path, jsonl_path


def load_index(index_dir: Path) -> tuple[np.ndarray, list[IndexRecord]]:
    """Read back what `build_index` wrote. Used by `retrieve.top_k`."""
    npy_path = index_dir / "embeddings.npy"
    jsonl_path = index_dir / "index.jsonl"
    if not npy_path.exists() or not jsonl_path.exists():
        raise FileNotFoundError(
            f"RAG index not found at {index_dir}. Run `civic-slm rag index <slug>` first."
        )
    arr = np.load(npy_path)
    records = [
        IndexRecord.from_dict(json.loads(line))
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return arr, records
