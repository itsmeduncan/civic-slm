"""`civic-slm review-sft` — terminal accept/reject loop for synthetic examples.

Schema validation catches structural problems, not whether the answer is
grounded in the context or whether the synth model fell into a repetitive
pattern. Curating the first ~500 by eye is the cheapest way to catch
systemic prompt-template problems before scaling.

Workflow:
  - reads `data/sft/{jurisdiction}.jsonl` (or `--input`)
  - shows each example with input/output rendered
  - accepts via [a]ccept / [r]eject / [s]kip / [q]uit
  - appends accepts to `data/sft/{jurisdiction}.curated.jsonl`
  - persists progress in `data/sft/.review_state.json` so you can resume
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from civic_slm.config import settings
from civic_slm.schema import InstructionExample

console = Console()


def _load(path: Path) -> list[InstructionExample]:
    return [
        InstructionExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _state_path(input_path: Path) -> Path:
    return input_path.parent / ".review_state.json"


def _load_state(input_path: Path) -> set[str]:
    p = _state_path(input_path)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("seen", []))


def _save_state(input_path: Path, seen: set[str]) -> None:
    _state_path(input_path).write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")


def main(
    jurisdiction: str = typer.Argument(
        None, help="Jurisdiction slug. Used to derive default --input / --out paths."
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="Override the synth output to review. Default: data/sft/{jurisdiction}.jsonl.",
    ),
    out_path: Path | None = typer.Option(
        None, "--out", help="Curated output. Default: data/sft/{jurisdiction}.curated.jsonl."
    ),
    limit: int = typer.Option(
        500, "--limit", "-n", help="Stop after reviewing this many new examples."
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override data directory (default: <repo>/data)."
    ),
) -> None:
    """Review synthetic SFT examples interactively.

    Examples:
      civic-slm review-sft san-clemente
      civic-slm review-sft san-clemente --limit 100
      civic-slm review-sft --input data/sft/custom.jsonl --out data/sft/custom.curated.jsonl
    """
    target_dir = data_dir or settings().data_dir

    if input_path is None:
        if not jurisdiction:
            raise typer.BadParameter(
                "Need either a jurisdiction argument or --input. "
                "Example: `civic-slm review-sft san-clemente`."
            )
        input_path = target_dir / "sft" / f"{jurisdiction}.jsonl"
    if out_path is None:
        stem = jurisdiction or input_path.stem
        out_path = target_dir / "sft" / f"{stem}.curated.jsonl"

    if not input_path.exists():
        typer.echo(f"input not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    examples = _load(input_path)
    seen = _load_state(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    reviewed = 0
    with out_path.open("a", encoding="utf-8") as out:
        for ex in examples:
            if ex.id in seen:
                continue
            if reviewed >= limit:
                break
            console.rule(f"[bold]{ex.task.value}[/bold] · {ex.id}")
            console.print(Panel(ex.input, title="input", border_style="cyan"))
            console.print(Panel(ex.output, title="output", border_style="green"))
            choice = Prompt.ask(
                "[a]ccept / [r]eject / [s]kip / [q]uit",
                choices=["a", "r", "s", "q"],
                default="a",
            )
            if choice == "q":
                break
            seen.add(ex.id)
            reviewed += 1
            if choice == "a":
                out.write(ex.model_dump_json() + "\n")
                accepted += 1
            elif choice == "s":
                seen.discard(ex.id)

    _save_state(input_path, seen)
    console.print(
        f"\n[bold]reviewed[/bold] {reviewed} · [green]accepted[/green] {accepted} -> {out_path}"
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        sys.exit(130)
