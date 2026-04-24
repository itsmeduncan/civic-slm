"""Round-trip + rejection tests for every public Pydantic model in schema.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from civic_slm.schema import (
    CivicDocument,
    DocType,
    DocumentChunk,
    EvalExample,
    EvalResult,
    ExtractionExample,
    FactualityExample,
    InstructionExample,
    PreferencePair,
    Provenance,
    RefusalExample,
    SideBySideExample,
    TaskType,
)

EVAL_ADAPTER: TypeAdapter[EvalExample] = TypeAdapter(EvalExample)


def _now() -> datetime:
    return datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def test_civic_document_roundtrip() -> None:
    doc = CivicDocument(
        id="ca/san-clemente/abc123def456",
        jurisdiction="san-clemente",
        state="CA",
        doc_type=DocType.AGENDA,
        source_url="https://example.gov/agenda.pdf",  # type: ignore[arg-type]
        retrieved_at=_now(),
        sha256="0" * 64,
        raw_path="data/raw/ca/san-clemente/2026-04-15/agenda.pdf",
        text="Item 1. Approval of minutes.",
    )
    blob = doc.model_dump_json()
    assert CivicDocument.model_validate_json(blob) == doc


def test_civic_document_rejects_bad_sha() -> None:
    with pytest.raises(ValidationError):
        CivicDocument(
            id="x",
            jurisdiction="x",
            state="CA",
            doc_type=DocType.OTHER,
            source_url="https://example.gov/x",  # type: ignore[arg-type]
            retrieved_at=_now(),
            sha256="not-a-real-sha",
            raw_path="x",
            text="x",
        )


def test_civic_document_rejects_bad_state() -> None:
    with pytest.raises(ValidationError):
        CivicDocument(
            id="x",
            jurisdiction="x",
            state="California",  # must be 2-letter postal code
            doc_type=DocType.OTHER,
            source_url="https://example.gov/x",  # type: ignore[arg-type]
            retrieved_at=_now(),
            sha256="0" * 64,
            raw_path="x",
            text="x",
        )


def test_document_chunk_roundtrip() -> None:
    chunk = DocumentChunk(
        doc_id="d1",
        chunk_idx=0,
        text="Section text.",
        token_count=42,
        section_path=["Land Use", "Goals"],
    )
    assert DocumentChunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_instruction_example_roundtrip() -> None:
    ex = InstructionExample(
        id="ex-001",
        task=TaskType.QA_GROUNDED,
        system="You are a civic assistant.",
        input="What was Item 3?",
        output="Item 3 was a budget amendment.",
        source_chunk_ids=["d1#0"],
        provenance=Provenance(
            generator="claude",
            model="claude-opus-4-7",
            prompt_sha="a" * 64,
            created_at=_now(),
        ),
    )
    assert InstructionExample.model_validate_json(ex.model_dump_json()) == ex


def test_preference_pair_roundtrip() -> None:
    pair = PreferencePair(
        id="p1",
        prompt="Summarize the agenda.",
        chosen="Three items: minutes, budget, public comment.",
        rejected="I don't know.",
        rationale="Chosen is grounded; rejected confabulates absence.",
    )
    assert PreferencePair.model_validate_json(pair.model_dump_json()) == pair


@pytest.mark.parametrize(
    "example",
    [
        FactualityExample(
            id="f1", question="Q?", context="C", gold_answer="A", gold_citations=["c1"]
        ),
        RefusalExample(id="r1", question="Q?", context="C"),
        ExtractionExample(
            id="e1",
            document_text="Doc",
            gold_json={"applicant": "Acme"},
            schema_name="staff_report",
        ),
        SideBySideExample(id="s1", prompt="P"),
    ],
)
def test_eval_example_discriminator_roundtrip(example: EvalExample) -> None:
    blob = EVAL_ADAPTER.dump_json(example)
    assert EVAL_ADAPTER.validate_json(blob) == example


def test_eval_example_rejects_unknown_bench() -> None:
    with pytest.raises(ValidationError):
        EVAL_ADAPTER.validate_python({"id": "x", "bench": "bogus"})


def test_eval_result_roundtrip() -> None:
    res = EvalResult(
        model_id="qwen-civic-sft-v1",
        bench="factuality",
        example_id="f1",
        prediction="42",
        score=0.85,
        judge_notes="close enough",
        latency_ms=120.5,
    )
    assert EvalResult.model_validate_json(res.model_dump_json()) == res


def test_eval_result_rejects_score_out_of_range() -> None:
    with pytest.raises(ValidationError):
        EvalResult(
            model_id="m",
            bench="factuality",
            example_id="f1",
            prediction="x",
            score=1.5,
            latency_ms=0.0,
        )
