"""`civic-slm prepare-cpt` — turn processed chunks into mlx_lm text-mode input.

CPT (`configs/cpt.yaml`) expects `data.format: text` with one JSON object per
line: `{"text": "..."}`. The processed-chunk JSONL we ship is a richer
`DocumentChunk` shape. This command projects the `text` field across one or
more jurisdictions into a single CPT corpus file.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.processed import load_chunks
from civic_slm.logging import configure, get_logger

log = get_logger(__name__)


def main(
    jurisdictions: list[str] = typer.Argument(
        None,
        help=(
            "One or more jurisdiction slugs to fold into the CPT corpus. "
            "Omit to use every processed JSONL on disk."
        ),
    ),
    out_path: Path = typer.Option(
        Path("data/processed/cpt.jsonl"),
        "--out",
        "-o",
        help='Output path. mlx_lm text-mode expects one `{"text": ...}` object per line.',
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override data directory (default: <repo>/data)."
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Shuffle chunks across jurisdictions before writing."
    ),
) -> None:
    """Project processed chunks into mlx_lm.lora text-mode corpus.

    Examples:
      civic-slm prepare-cpt san-clemente
      civic-slm prepare-cpt san-clemente santa-monica --out data/processed/two-cities.jsonl
      civic-slm prepare-cpt                 # all jurisdictions found on disk
    """
    configure()
    target_dir = data_dir or settings().data_dir

    if not jurisdictions:
        processed = target_dir / "processed"
        jurisdictions = sorted(
            p.stem for p in processed.glob("*.jsonl") if p.name not in {"cpt.jsonl"}
        )
        if not jurisdictions:
            raise typer.BadParameter(
                f"No processed JSONLs found under {processed}. "
                f"Run `civic-slm process <jurisdiction>` first."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_texts: list[str] = []
    for j in jurisdictions:
        chunks = load_chunks(j, data_dir=target_dir)
        if not chunks:
            log.warning("no_processed_chunks", jurisdiction=j)
            continue
        all_texts.extend(c.text for c in chunks)
        log.info("loaded_jurisdiction", jurisdiction=j, chunks=len(chunks))

    if not all_texts:
        raise typer.BadParameter(
            "No chunks loaded. Did `civic-slm process <jurisdiction>` actually run?"
        )

    if shuffle:
        import random

        random.shuffle(all_texts)

    with out_path.open("w", encoding="utf-8") as fh:
        for text in all_texts:
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    typer.echo(
        f"Wrote {len(all_texts)} text records from "
        f"{len(jurisdictions)} jurisdiction(s) -> {out_path}"
    )


if __name__ == "__main__":
    typer.run(main)
