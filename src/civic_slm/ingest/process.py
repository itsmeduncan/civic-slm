from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError

from civic_slm.config import settings
from civic_slm.ingest.manifest import load_manifest
from civic_slm.ingest.pdf import chunk_text, extract_pdf
from civic_slm.ingest.processed import save_chunks
from civic_slm.logging import configure, get_logger


def _pdf_read_error_class() -> type[Exception]:
    """Resolve `pypdf.errors.PdfReadError` lazily.

    `pypdf` lives in the `ingest` extra; this module must still import
    cleanly without it (CI runs `synth + eval` only on Linux). We resolve
    the class once on first call and cache via a closure on the function.
    """
    try:
        from pypdf.errors import PdfReadError  # type: ignore[import-not-found]
    except ImportError:
        # Sentinel: a private subclass that can never be raised. Catching
        # it is a no-op, so the surrounding `(OSError, ValidationError, ...)`
        # behaves correctly when pypdf isn't installed.
        class _Unreachable(Exception):
            pass

        return _Unreachable
    return PdfReadError


log = get_logger(__name__)


def main(
    jurisdiction: str = typer.Argument(..., help="Jurisdiction to process (e.g., san-clemente)"),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
) -> None:
    """Process raw crawled PDFs into section-aware chunks for training/synth.

    Reads from the manifest and writes to data/processed/{jurisdiction}.jsonl.
    """
    configure()
    target = data_dir or settings().data_dir
    manifest = load_manifest(target)

    docs = [d for d in manifest if d.jurisdiction == jurisdiction]
    if not docs:
        typer.echo(
            f"No documents found in manifest for {jurisdiction}. Please crawl first.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Processing {len(docs)} documents for {jurisdiction}...")

    all_chunks = []
    for doc in docs:
        # raw_path is relative to data_dir and already includes the "raw/" prefix.
        pdf_path = target / doc.raw_path
        if not pdf_path.exists():
            log.warning("file_missing", path=str(pdf_path))
            continue

        try:
            pages = extract_pdf(pdf_path)
            full_text = "\n\n".join([p.text for p in pages])

            chunks = chunk_text(
                doc_id=doc.sha256,
                text=full_text,
                source_doc_hash=doc.sha256,
            )
            all_chunks.extend(chunks)
            log.info("doc_processed", doc=doc.raw_path, chunks=len(chunks))
        except (OSError, ValidationError, _pdf_read_error_class()) as e:
            # A single malformed PDF or chunk-validation failure shouldn't
            # kill the whole batch. Programming errors propagate so we
            # actually see them in tests. pypdf is lazy-loaded so this
            # module imports without the `ingest` extra installed.
            log.error("process_failed", path=str(pdf_path), error=str(e))

    save_chunks(jurisdiction, all_chunks)

    typer.echo(f"Successfully processed {len(docs)} docs into {len(all_chunks)} chunks.")
    typer.echo(f"Saved to data/processed/{jurisdiction}.jsonl")


if __name__ == "__main__":
    typer.run(main)
