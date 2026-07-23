"""`civic-slm curate` — model-driven first pass over a synth SFT corpus.

Scores + defect-classifies each InstructionExample (cheap Haiku call), then
routes it to accept / human-queue / reject. See
docs/superpowers/specs/2026-07-23-synth-curator-design.md.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

from civic_slm.config import settings
from civic_slm.jsonparse import extract_first
from civic_slm.llm.backend import Backend, select_backend
from civic_slm.logging import configure, get_logger
from civic_slm.schema import (
    HIGH_SEVERITY,
    CurationVerdict,
    DefectClass,
    InstructionExample,
    QueuedExample,
)

log = get_logger(__name__)
console = Console()
_PROMPT = (Path(__file__).parent / "prompts" / "curate.txt").read_text(encoding="utf-8")


class Bucket(StrEnum):
    ACCEPT = "accept"
    QUEUE = "queue"
    REJECT = "reject"


def disposition(v: CurationVerdict) -> Bucket:
    """Route a verdict to a bucket.

    Conservative on ACCEPT: any defect (even a fixable one) sends the example to
    the human queue so the suggested fix is seen. A false auto-accept trains on
    bad data; a false queue only costs review time.
    """
    if any(d in HIGH_SEVERITY for d in v.defects) or v.score <= 3:
        return Bucket.REJECT
    if v.score >= 8 and not v.defects:
        return Bucket.ACCEPT
    return Bucket.QUEUE


def parse_verdict(text: str, example_id: str) -> CurationVerdict | None:
    """Parse a curator's raw reply into a CurationVerdict, or None if unusable."""
    obj, status = extract_first(text, "object")
    if status != "ok" or not isinstance(obj, dict):
        return None
    try:
        return CurationVerdict.model_validate({**obj, "example_id": example_id})
    except ValueError:
        return None


async def curate_example(ex: InstructionExample, backend: Backend) -> CurationVerdict:
    """Curate one example. Never raises: any failure yields a QUEUE-bound verdict."""
    user = _PROMPT.format(task=ex.task.value, input=ex.input, output=ex.output)
    try:
        text = await backend.complete(system=None, user=user, max_tokens=512)
    except Exception as exc:  # network / SDK / provider error — fail safe to queue
        log.warning("curate_backend_error", example_id=ex.id, error=str(exc))
        text = ""
    verdict = parse_verdict(text, ex.id)
    if verdict is None:
        log.warning("curate_unparseable", example_id=ex.id)
        # score 5 + no defects -> disposition() routes to QUEUE (never dropped/accepted)
        return CurationVerdict(
            example_id=ex.id,
            score=5,
            defects=[],
            rationale="curator output unparseable; queued for human",
        )
    return verdict


@dataclass
class CurateSummary:
    accept: int = 0
    queue: int = 0
    reject: int = 0
    defects: dict[str, int] = field(default_factory=dict)


def _state_path(out_dir: Path, stem: str) -> Path:
    """Resume-state file, co-located with the split outputs it tracks.

    Keying on the out-dir (not the input) is deliberate: the `seen` set records
    which examples have already been *written to this out-dir*. Keying on the
    input instead meant a second run with a different `--out-dir` read the same
    state, saw every example as done, and silently wrote nothing to the new
    out-dir. For the default `--out-dir data/sft` this resolves to the same path
    as before, so existing state files keep working.
    """
    return out_dir / f"{stem}.curate-state.json"


def _load_seen(out_dir: Path, stem: str) -> set[str]:
    p = _state_path(out_dir, stem)
    return set(json.loads(p.read_text()).get("seen", [])) if p.exists() else set()


_PII_REDACTION = "[redacted: pii_leak]"


def _redact_for_storage(ex: InstructionExample, verdict: CurationVerdict) -> InstructionExample:
    """Scrub the example body before persisting a `pii_leak` reject.

    Rejects are written verbatim to `{stem}.rejected.jsonl` for audit, but for a
    `pii_leak` verdict that body *is* the leaked PII. Store a redacted copy — the
    id, task, provenance, and full verdict survive for audit, only the `input` /
    `output` content fields are blanked, so the PII never lands on disk. A no-op
    for every other reject reason.
    """
    if DefectClass.PII_LEAK not in verdict.defects:
        return ex
    return ex.model_copy(update={"input": _PII_REDACTION, "output": _PII_REDACTION})


async def curate_corpus(
    *,
    input_path: Path,
    out_dir: Path,
    backend: Backend,
    concurrency: int = 8,
    limit: int | None = None,
) -> CurateSummary:
    """Curate every example in input_path; append to the 3 split files. Resumable."""
    stem = input_path.stem
    curated_p = out_dir / f"{stem}.curated.jsonl"
    queue_p = out_dir / f"{stem}.curate-queue.jsonl"
    # `.rejected.jsonl` holds rejects for audit. It lives under gitignored
    # data/sft/, so it is never committed by accident — do not `git add -f` it,
    # and note pii_leak reject bodies are redacted before they land here.
    reject_p = out_dir / f"{stem}.rejected.jsonl"
    seen = _load_seen(out_dir, stem)

    examples = [
        InstructionExample.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    todo = [e for e in examples if e.id not in seen]
    if limit is not None:
        todo = todo[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def one(ex: InstructionExample) -> tuple[InstructionExample, CurationVerdict]:
        async with sem:
            return ex, await curate_example(ex, backend)

    results = await asyncio.gather(*(one(e) for e in todo))

    counts = Counter[str]()
    defect_hist = Counter[str]()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        curated_p.open("a", encoding="utf-8") as fa,
        queue_p.open("a", encoding="utf-8") as fq,
        reject_p.open("a", encoding="utf-8") as fr,
    ):
        for ex, verdict in results:
            for d in verdict.defects:
                defect_hist[d.value] += 1
            bucket = disposition(verdict)
            counts[bucket.value] += 1
            if bucket is Bucket.ACCEPT:
                fa.write(ex.model_dump_json() + "\n")
            elif bucket is Bucket.QUEUE:
                fq.write(QueuedExample(example=ex, verdict=verdict).model_dump_json() + "\n")
            else:  # REJECT — scrub PII before it is persisted for audit
                stored = _redact_for_storage(ex, verdict)
                fr.write(QueuedExample(example=stored, verdict=verdict).model_dump_json() + "\n")
            seen.add(ex.id)

    _state_path(out_dir, stem).write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
    return CurateSummary(
        accept=counts["accept"],
        queue=counts["queue"],
        reject=counts["reject"],
        defects=dict(defect_hist),
    )


def main(
    slug: str = typer.Argument(
        ..., help="Jurisdiction slug (default input data/sft/{slug}.jsonl)."
    ),
    input_path: Path | None = typer.Option(
        None, "--input", help="Override the synth jsonl to curate."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", help="Where the 3 split files land. Default: data/sft/."
    ),
    model: str = typer.Option(
        "claude-haiku-4-5", "--model", envvar="CIVIC_SLM_CURATOR_MODEL", help="Curator model."
    ),
    concurrency: int = typer.Option(
        8, "--concurrency", min=1, help="Max concurrent curator calls."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Curate at most N examples this run."),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Override the project data dir."),
) -> None:
    """Score + defect-classify a synth SFT corpus; split into accept/queue/reject."""
    configure()
    base = data_dir or settings().data_dir
    inp = input_path or base / "sft" / f"{slug}.jsonl"
    if not inp.exists():
        raise typer.BadParameter(
            f"no synth corpus at {inp} — run `civic-slm synth {slug}` first, or pass --input."
        )
    out = out_dir or base / "sft"
    backend = select_backend(default_anthropic_model=model)
    summary = asyncio.run(
        curate_corpus(
            input_path=inp, out_dir=out, backend=backend, concurrency=concurrency, limit=limit
        )
    )
    console.print(
        f"[bold]curated {slug}[/bold]: "
        f"[green]{summary.accept} accept[/green] · "
        f"[yellow]{summary.queue} queue[/yellow] · [red]{summary.reject} reject[/red]"
    )
    if summary.defects:
        console.print(
            "defects: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.defects.items()))
        )
    console.print(f"→ review the queue: [cyan]civic-slm review-sft {slug} --queue[/cyan]")
