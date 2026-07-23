from __future__ import annotations

import pytest

from civic_slm.schema import CurationVerdict, DefectClass
from civic_slm.synth.curate import Bucket, disposition


def _v(score: int, defects: list[DefectClass] | None = None) -> CurationVerdict:
    return CurationVerdict(example_id="e1", score=score, defects=defects or [], rationale="r")


@pytest.mark.parametrize(
    "score,defects,expected",
    [
        (10, [], Bucket.ACCEPT),
        (8, [], Bucket.ACCEPT),
        (9, [DefectClass.FORMAT_DRIFT], Bucket.QUEUE),  # any defect blocks auto-accept
        (7, [], Bucket.QUEUE),
        (4, [], Bucket.QUEUE),
        (3, [], Bucket.REJECT),
        (0, [], Bucket.REJECT),
        (10, [DefectClass.PII_LEAK], Bucket.REJECT),  # high-severity overrides score
        (10, [DefectClass.UNGROUNDED_ANSWER], Bucket.REJECT),
        (10, [DefectClass.CONFUSED_REFUSAL], Bucket.REJECT),
        (6, [DefectClass.SCHEMA_DRIFT], Bucket.QUEUE),  # fixable defect -> queue
    ],
)
def test_disposition(score, defects, expected):
    assert disposition(_v(score, defects)) is expected


def test_verdict_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        CurationVerdict(example_id="e", score=11, rationale="r")
