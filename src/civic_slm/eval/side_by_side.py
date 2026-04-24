"""Side-by-side runner: candidate vs comparator, judged pairwise by Claude.

Score per example: 1.0 if candidate wins, 0.5 if tie, 0.0 if comparator wins.
Aggregate mean is the candidate's win rate (with ties at 0.5 — standard ELO-like
convention). Position bias controlled in `judge.judge_with_position_swap`.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import TypeAdapter

from civic_slm.config import settings
from civic_slm.eval.judge import judge_with_position_swap
from civic_slm.eval.runner import write_report
from civic_slm.logging import configure, get_logger
from civic_slm.schema import EvalExample, EvalResult, SideBySideExample
from civic_slm.serve.client import ChatClient

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)
app = typer.Typer(help="Run pairwise side_by_side evaluation.")
EVAL_ADAPTER: TypeAdapter[EvalExample] = TypeAdapter(EvalExample)

_SYSTEM = "You are a helpful assistant on California municipal government."


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped


def _load(path: Path) -> list[SideBySideExample]:
    examples = [EVAL_ADAPTER.validate_json(line) for line in _iter_lines(path)]
    return [ex for ex in examples if isinstance(ex, SideBySideExample)]


def run_side_by_side(
    *,
    examples: list[SideBySideExample],
    candidate: ChatClient,
    comparator: ChatClient,
    candidate_id: str,
    judge_model: str = "claude-sonnet-4-6",
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for ex in examples:
        cand = candidate.chat(_SYSTEM, ex.prompt)
        comp = comparator.chat(_SYSTEM, ex.prompt)
        verdict = judge_with_position_swap(
            prompt=ex.prompt,
            rubric=ex.rubric or "general quality",
            response_a=cand.text,
            response_b=comp.text,
            model=judge_model,
        )
        score = {"A": 1.0, "tie": 0.5, "B": 0.0}[verdict.winner]
        results.append(
            EvalResult(
                model_id=candidate_id,
                bench="side_by_side",
                example_id=ex.id,
                prediction=cand.text,
                score=score,
                judge_notes=f"verdict={verdict.winner}; {verdict.reason}",
                latency_ms=cand.latency_ms,
            )
        )
        log.info("side_by_side", id=ex.id, winner=verdict.winner)
    return results


@app.command()
def main(
    candidate_model: str = typer.Option(..., help="Candidate model id (artifact dir)."),
    bench_file: Path = typer.Option(Path("data/eval/side_by_side.jsonl")),
    candidate_url: str = typer.Option("http://localhost:8080", help="Candidate server URL."),
    comparator_url: str = typer.Option(
        "http://localhost:8081", help="Comparator (e.g. 72B llama-server) URL."
    ),
    candidate_served: str = typer.Option("default", help="Server-side model name for candidate."),
    comparator_served: str = typer.Option("default", help="Server-side model name for comparator."),
    judge_model: str = typer.Option("claude-sonnet-4-6", help="Anthropic judge model."),
) -> None:
    configure()
    examples = _load(bench_file)
    log.info("loaded_side_by_side", count=len(examples))
    cand = ChatClient(base_url=candidate_url, model=candidate_served)
    comp = ChatClient(base_url=comparator_url, model=comparator_served)
    results = run_side_by_side(
        examples=examples,
        candidate=cand,
        comparator=comp,
        candidate_id=candidate_model,
        judge_model=judge_model,
    )
    out_dir = settings().artifacts_dir / "evals" / candidate_model
    write_report(results, out_dir, "side_by_side")
    if results:
        win_rate = statistics.mean(r.score for r in results)
        typer.echo(f"win_rate={win_rate:.3f}")


if __name__ == "__main__":
    app()
