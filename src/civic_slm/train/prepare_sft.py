"""`civic-slm prepare-sft` — curated InstructionExamples → chat-format SFT data.

Replaces the standalone `scripts/prepare_sft.py`. Adds a train/valid split
(default 90/10) so the output matches what `configs/sft.yaml` expects out of
the box.

Why kept separate from synth: synth writes the rich `InstructionExample`
schema so we can re-derive chat records later (after prompt-template tweaks)
without re-calling the teacher. The chat-format conversion is cheap and
deterministic; this CLI is the deterministic step.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.logging import configure, get_logger
from civic_slm.schema import InstructionExample

log = get_logger(__name__)


def main(
    input_path: Path = typer.Argument(
        ..., help="Curated or raw `InstructionExample` JSONL (output of synth or review-sft)."
    ),
    out_dir: Path = typer.Option(
        Path("data/sft/sft"),
        "--out-dir",
        "-o",
        help=(
            "Output directory. mlx_lm.lora --data <dir> expects `train.jsonl` "
            "(+ optional `valid.jsonl`) in this directory, matching the "
            "prepare-cpt convention. Default lives under data/sft/ so the "
            "raw synth JSONL stays as the source of truth."
        ),
    ),
    valid_ratio: float = typer.Option(
        0.1, "--valid-ratio", min=0.0, max=0.5, help="Fraction of records held out for validation."
    ),
    seed: int = typer.Option(
        17, help="Shuffle seed so train/valid splits are reproducible across runs."
    ),
) -> None:
    """Split curated examples into mlx_lm-shaped train.jsonl + valid.jsonl.

    Examples:
      civic-slm prepare-sft data/sft/san-clemente.jsonl
      civic-slm prepare-sft data/sft/v0.curated.jsonl --valid-ratio 0.05
    """
    configure()

    if not input_path.exists():
        raise typer.BadParameter(f"Input not found: {input_path}")

    examples: list[InstructionExample] = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        examples.append(InstructionExample.model_validate_json(line))

    if not examples:
        raise typer.BadParameter(f"No valid records in {input_path}.")

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_valid = max(1, int(len(examples) * valid_ratio)) if valid_ratio > 0 else 0
    valid = examples[:n_valid]
    train = examples[n_valid:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    for path, batch in ((train_path, train), (valid_path, valid)):
        with path.open("w", encoding="utf-8") as fh:
            for ex in batch:
                fh.write(json.dumps(ex.to_chat_record(), ensure_ascii=False) + "\n")

    typer.echo(
        f"Wrote {len(train)} train + {len(valid)} valid chat-format records "
        f"({input_path} -> {out_dir})"
    )

    # Convenience: surface the SFT-config pointer the user usually edits next.
    settings()  # touch settings to keep import side-effects consistent
    typer.echo(
        "Configure `configs/sft.yaml` so:\n"
        f"  data.train_path: {train_path}\n"
        f"  data.valid_path: {valid_path}"
    )


if __name__ == "__main__":
    typer.run(main)
