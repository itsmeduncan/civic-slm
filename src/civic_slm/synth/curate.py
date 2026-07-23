"""`civic-slm curate` — model-driven first pass over a synth SFT corpus.

Scores + defect-classifies each InstructionExample (cheap Haiku call), then
routes it to accept / human-queue / reject. See
docs/superpowers/specs/2026-07-23-synth-curator-design.md.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from civic_slm.jsonparse import extract_first
from civic_slm.llm.backend import Backend
from civic_slm.logging import get_logger
from civic_slm.schema import HIGH_SEVERITY, CurationVerdict, InstructionExample, QueuedExample

log = get_logger(__name__)
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
    defects: dict[str, int] | None = None


def _state_path(input_path: Path) -> Path:
    return input_path.with_suffix(".curate-state.json")


def _load_seen(input_path: Path) -> set[str]:
    p = _state_path(input_path)
    return set(json.loads(p.read_text()).get("seen", [])) if p.exists() else set()


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
    reject_p = out_dir / f"{stem}.rejected.jsonl"
    seen = _load_seen(input_path)

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
            else:
                target = fq if bucket is Bucket.QUEUE else fr
                target.write(QueuedExample(example=ex, verdict=verdict).model_dump_json() + "\n")
            seen.add(ex.id)

    _state_path(input_path).write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
    return CurateSummary(
        accept=counts["accept"],
        queue=counts["queue"],
        reject=counts["reject"],
        defects=dict(defect_hist),
    )
