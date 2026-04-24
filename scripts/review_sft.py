"""Terminal accept/reject CLI for synthetic SFT examples.

Why human review at all when validation already runs: schema validation only
catches structural problems. It can't tell you whether the *answer* is grounded
in the *context*, whether the question is leading, or whether the synth model
fell into a repetitive pattern. Curating the first ~500 by eye is the cheapest
way to catch systemic prompt-template problems before scaling to 10k examples.

Workflow:
  - reads `data/sft/v0.jsonl`
  - shows each example with input/output rendered
  - accepts via [a]ccept / [r]eject / [s]kip / [q]uit
  - appends accepts to `data/sft/v0.curated.jsonl`
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

from civic_slm.schema import InstructionExample

app = typer.Typer(help="Review synthetic SFT examples.", no_args_is_help=False)
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


@app.command()
def main(
    input_path: Path = typer.Option(Path("data/sft/v0.jsonl"), help="Input JSONL."),
    out_path: Path = typer.Option(Path("data/sft/v0.curated.jsonl"), help="Curated output."),
    limit: int = typer.Option(500, help="Stop after reviewing this many new examples."),
) -> None:
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
                seen.discard(ex.id)  # don't mark — review again later

    _save_state(input_path, seen)
    console.print(f"\n[bold]reviewed[/bold] {reviewed} · [green]accepted[/green] {accepted}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
