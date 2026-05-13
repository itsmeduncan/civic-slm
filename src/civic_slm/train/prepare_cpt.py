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
    out_dir: Path = typer.Option(
        Path("data/processed/cpt"),
        "--out-dir",
        "-o",
        help=(
            "Output directory. mlx_lm.lora text-mode expects `train.jsonl` "
            "(+ optional `valid.jsonl`) in this directory; we split 90/10 unless "
            "there's only one chunk, in which case the single line goes into both."
        ),
    ),
    valid_ratio: float = typer.Option(
        0.1,
        help=(
            "Fraction of chunks to hold out for validation. Floor of one chunk "
            "in `valid.jsonl` unless the input corpus has zero chunks."
        ),
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

    out_dir.mkdir(parents=True, exist_ok=True)

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

    # 90/10 split with floors: validation always gets ≥ 1 line if we have ≥ 2;
    # with a single-chunk corpus the same line goes into both so mlx_lm can
    # still report eval loss during a smoke test.
    n_valid = max(1, int(len(all_texts) * valid_ratio)) if len(all_texts) >= 2 else 1
    valid_texts = all_texts[:n_valid]
    train_texts = all_texts[n_valid:] if len(all_texts) >= 2 else all_texts

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    for path, texts in ((train_path, train_texts), (valid_path, valid_texts)):
        with path.open("w", encoding="utf-8") as fh:
            for text in texts:
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    typer.echo(
        f"Wrote {len(train_texts)} train + {len(valid_texts)} valid records "
        f"from {len(jurisdictions)} jurisdiction(s) -> {out_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
