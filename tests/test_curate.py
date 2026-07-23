from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import dataclass as _dc
from datetime import UTC, datetime
from pathlib import Path

import pytest

from civic_slm.schema import (
    CurationVerdict,
    DefectClass,
    InstructionExample,
    Provenance,
    QueuedExample,
    TaskType,
)
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


def _example() -> InstructionExample:
    return InstructionExample(
        id="ex-1",
        task=TaskType.QA_GROUNDED,
        system="sys",
        input="Context:\nItem 8A raises water rates.",
        output="Item 8A raises water rates.",
        source_chunk_ids=["c1"],
        provenance=Provenance(
            prompt_sha="a" * 64,
            model="claude",
            generator="claude",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def test_parse_verdict_clean_json():
    from civic_slm.synth.curate import parse_verdict

    raw = json.dumps({"score": 9, "defects": [], "suggested_fix": None, "rationale": "grounded"})
    v = parse_verdict(raw, "ex-1")
    assert v is not None and v.score == 9 and v.example_id == "ex-1" and v.defects == []


def test_parse_verdict_json_in_prose():
    from civic_slm.synth.curate import parse_verdict

    raw = (
        'Here is my assessment:\n```json\n{"score": 5, "defects": '
        '["leading_question"], "rationale": "leads"}\n```'
    )
    v = parse_verdict(raw, "ex-1")
    assert v is not None and v.score == 5 and v.defects[0].value == "leading_question"


def test_parse_verdict_malformed_returns_none():
    from civic_slm.synth.curate import parse_verdict

    assert parse_verdict("not json at all", "ex-1") is None


@pytest.mark.asyncio
async def test_curate_example_failsafe_on_bad_backend():
    from civic_slm.synth.curate import curate_example

    @dataclass
    class BadBackend:
        model: str = "claude-haiku-4-5"

        async def complete(self, *, system, user, max_tokens=4096) -> str:
            return "garbage, no json"

    v = await curate_example(_example(), BadBackend())
    assert v.example_id == "ex-1" and disposition(v) is Bucket.QUEUE  # fail-safe -> queue


@_dc
class StubBackend:
    """Scores by a marker in the output text so we can drive each bucket."""

    model: str = "claude-haiku-4-5"

    async def complete(self, *, system, user, max_tokens=4096) -> str:
        if "PII" in user:
            return '{"score": 10, "defects": ["pii_leak"], "rationale": "name"}'
        if "GOOD" in user:
            return '{"score": 9, "defects": [], "rationale": "clean"}'
        return (
            '{"score": 6, "defects": ["format_drift"], '
            '"suggested_fix": "use prose", "rationale": "bullets"}'
        )


def _ex(i: str, out: str) -> InstructionExample:
    return InstructionExample(
        id=i,
        task=TaskType.QA_GROUNDED,
        system="s",
        input="c",
        output=out,
        source_chunk_ids=["c1"],
        provenance=Provenance(
            prompt_sha="a" * 64,
            model="m",
            generator="claude",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


@pytest.mark.asyncio
async def test_curate_corpus_splits_and_summarizes(tmp_path: Path):
    from civic_slm.synth.curate import curate_corpus

    inp = tmp_path / "san.jsonl"
    inp.write_text(
        "\n".join(
            e.model_dump_json() for e in [_ex("g1", "GOOD"), _ex("p1", "PII"), _ex("q1", "meh")]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = await curate_corpus(
        input_path=inp, out_dir=tmp_path, backend=StubBackend(), concurrency=2
    )
    assert (summary.accept, summary.queue, summary.reject) == (1, 1, 1)
    assert summary.defects["pii_leak"] == 1 and summary.defects["format_drift"] == 1
    assert (tmp_path / "san.curated.jsonl").read_text().count("g1") == 1
    q = [
        QueuedExample.model_validate_json(line)
        for line in (tmp_path / "san.curate-queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert q[0].example.id == "q1" and q[0].verdict.suggested_fix == "use prose"
    # resumable: a second run over the same input adds nothing new
    s2 = await curate_corpus(input_path=inp, out_dir=tmp_path, backend=StubBackend(), concurrency=2)
    assert (s2.accept, s2.queue, s2.reject) == (0, 0, 0)


def test_cli_curate_runs(tmp_path: Path, monkeypatch):
    from civic_slm.synth import curate as curate_mod

    inp = tmp_path / "san.jsonl"
    inp.write_text(_ex("g1", "GOOD").model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(curate_mod, "select_backend", lambda **_: StubBackend())
    curate_mod.main(
        slug="san",
        input_path=inp,
        out_dir=tmp_path,
        model="claude-haiku-4-5",
        concurrency=2,
        limit=None,
        data_dir=tmp_path,
    )
    assert (tmp_path / "san.curated.jsonl").exists()
