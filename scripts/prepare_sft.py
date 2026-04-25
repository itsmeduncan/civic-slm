"""Convert curated `InstructionExample` JSONL to the chat format mlx_lm.lora expects.

Reads `data/sft/v0.curated.jsonl` (schema-rich records), writes
`data/sft/v0.train.jsonl` as `{"messages": [{role, content}, ...]}` per line.
This is the file `configs/sft.yaml` points at via `data.train_path`.

Why a separate step rather than writing chat format directly from `synth`:
keeping the rich schema around in `v0.curated.jsonl` lets us re-derive chat
records later (e.g. after prompt-template changes) without re-calling the
teacher. The conversion is cheap and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from civic_slm.schema import InstructionExample

app = typer.Typer(help="Convert curated SFT examples to chat-format JSONL.")


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/sft/v0.curated.jsonl"), help="Curated InstructionExample JSONL."
    ),
    output_path: Path = typer.Option(
        Path("data/sft/v0.train.jsonl"), help="Chat-format output JSONL."
    ),
) -> None:
    if not input_path.exists():
        typer.echo(f"input not found: {input_path}", err=True)
        raise typer.Exit(code=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as out:
        for raw in input_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            ex = InstructionExample.model_validate_json(line)
            out.write(json.dumps(ex.to_chat_record()) + "\n")
            n += 1
    typer.echo(f"wrote {n} chat-format records → {output_path}")


if __name__ == "__main__":
    app()
