"""`civic-slm curate` — model-driven first pass over a synth SFT corpus.

Scores + defect-classifies each InstructionExample (cheap Haiku call), then
routes it to accept / human-queue / reject. See
docs/superpowers/specs/2026-07-23-synth-curator-design.md.
"""

from __future__ import annotations

from enum import StrEnum

from civic_slm.schema import HIGH_SEVERITY, CurationVerdict


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
