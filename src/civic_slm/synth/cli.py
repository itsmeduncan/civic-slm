"""Typer wrapper around `civic_slm.synth.generate.generate_corpus`.

This is the user-facing command that turns processed chunks into synthetic
SFT pairs. It hides the asyncio entry-point boilerplate and the
chunk/state/doc-type plumbing.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.manifest import load_manifest
from civic_slm.ingest.processed import load_chunks
from civic_slm.logging import configure, get_logger
from civic_slm.schema import DocType, TaskType
from civic_slm.synth.generate import generate_corpus

log = get_logger(__name__)


_DEFAULT_TASKS: tuple[TaskType, ...] = (
    TaskType.QA_GROUNDED,
    TaskType.REFUSAL,
    TaskType.EXTRACT,
    TaskType.SUMMARIZE,
)


def _resolve_state(jurisdiction: str, data_dir: Path, override: str | None) -> str:
    """Look up the 2-letter state for `jurisdiction` from the manifest.

    Falls back to `override` (the `--state` CLI flag) when the manifest is
    missing or doesn't have an entry for this jurisdiction — useful when
    processed chunks survived a `git clean` but the manifest didn't.
    """
    if override is not None:
        return override.upper()
    try:
        manifest = load_manifest(data_dir)
    except FileNotFoundError:
        manifest = []
    states = {d.state for d in manifest if d.jurisdiction == jurisdiction}
    if not states:
        raise typer.BadParameter(
            f"No manifest entries for jurisdiction={jurisdiction!r} and no "
            f"--state override given. Either crawl first "
            f"(`civic-slm crawl --jurisdiction {jurisdiction}`) or pass "
            f"`--state CA` to bypass the manifest lookup."
        )
    if len(states) > 1:
        raise typer.BadParameter(
            f"Manifest entries for {jurisdiction!r} disagree on state: {states}. "
            f"Fix the manifest or pass --state explicitly."
        )
    return states.pop()


def _resolve_doc_type(jurisdiction: str, data_dir: Path, override: DocType | None) -> DocType:
    """Pick the dominant doc_type from the manifest, or use the override."""
    if override is not None:
        return override
    try:
        manifest = load_manifest(data_dir)
    except FileNotFoundError:
        manifest = []
    counts = Counter(d.doc_type for d in manifest if d.jurisdiction == jurisdiction)
    if not counts:
        return DocType.OTHER
    most_common, _ = counts.most_common(1)[0]
    return most_common


def main(
    jurisdiction: str = typer.Argument(..., help="Jurisdiction slug (e.g., san-clemente)."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output JSONL path. Default: data/sft/{jurisdiction}.jsonl.",
    ),
    n_per_chunk: int = typer.Option(
        3, "--n-per-chunk", "-n", min=1, help="Examples per (chunk, task)."
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", "-c", min=1, help="Max concurrent backend calls."
    ),
    tasks: list[TaskType] = typer.Option(
        None,
        "--task",
        "-t",
        help="Task to generate (repeatable). Default: qa_grounded, refusal, extract, summarize.",
    ),
    doc_type: DocType | None = typer.Option(
        None,
        "--doc-type",
        help="Override doc_type used in prompts. Default: dominant doc_type in manifest.",
    ),
    state: str | None = typer.Option(
        None,
        "--state",
        help="2-letter U.S. state, e.g. CA. Required only when the manifest is unavailable.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help=(
            "Skip (chunk, task, round) triples already present in --out. "
            "Use --no-resume to force re-run starting at round 0."
        ),
    ),
    rounds: int = typer.Option(
        1,
        "--rounds",
        "-r",
        min=1,
        help=(
            "How many synth passes to run. With --resume (default), rounds "
            "stack on top of any existing rounds on disk — so `--rounds 4` "
            "against a file already containing round 0 generates rounds 1-4. "
            "Useful when you need more examples per (chunk, task) than fit "
            "in a single Claude completion."
        ),
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override data directory (default: <repo>/data)."
    ),
) -> None:
    """Generate synthetic SFT pairs from processed chunks.

    Reads chunks from `data/processed/{jurisdiction}.jsonl` (produced by
    `civic-slm process`) and writes instruction examples to
    `data/sft/{jurisdiction}.jsonl` by default. The backend is chosen from
    `CIVIC_SLM_LLM_BACKEND` (`anthropic` or `local`) — see docs/RUNTIMES.md.
    """
    configure()

    target_dir = data_dir or settings().data_dir
    chunks = load_chunks(jurisdiction, data_dir=target_dir)
    if not chunks:
        raise typer.BadParameter(
            f"No processed chunks for {jurisdiction!r}. "
            f"Run `civic-slm process {jurisdiction}` first."
        )

    resolved_state = _resolve_state(jurisdiction, target_dir, state)
    resolved_doc_type = _resolve_doc_type(jurisdiction, target_dir, doc_type)
    out_path = out or (target_dir / "sft" / f"{jurisdiction}.jsonl")
    chosen_tasks: tuple[TaskType, ...] = tuple(tasks) if tasks else _DEFAULT_TASKS

    typer.echo(
        f"Synthesizing {len(chunks)} chunks x {len(chosen_tasks)} tasks x "
        f"{n_per_chunk} examples x {rounds} round(s) -> {out_path}"
    )

    total = asyncio.run(
        generate_corpus(
            chunks=chunks,
            jurisdiction=jurisdiction,
            state=resolved_state,
            doc_type=resolved_doc_type.value,
            out_path=out_path,
            n_per_chunk=n_per_chunk,
            tasks=chosen_tasks,
            concurrency=concurrency,
            resume=resume,
            rounds=rounds,
        )
    )

    typer.echo(f"Wrote {total} examples to {out_path}")


if __name__ == "__main__":
    typer.run(main)
