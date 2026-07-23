"""`civic-slm curate` — model-driven first pass over a synth SFT corpus.

Scores + defect-classifies each InstructionExample (cheap Haiku call), then
routes it to accept / human-queue / reject. See
docs/superpowers/specs/2026-07-23-synth-curator-design.md.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from civic_slm.jsonparse import extract_first
from civic_slm.llm.backend import Backend
from civic_slm.logging import get_logger
from civic_slm.schema import HIGH_SEVERITY, CurationVerdict, InstructionExample

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
