"""Tests for the synth parser. The Anthropic call itself is not exercised here."""

from __future__ import annotations

from datetime import UTC, datetime

from civic_slm.schema import Provenance, TaskType
from civic_slm.synth.generate import parse_examples


def _provenance() -> Provenance:
    return Provenance(
        generator="claude",
        model="claude-opus-4-7",
        prompt_sha="a" * 64,
        created_at=datetime(2026, 4, 24, tzinfo=UTC),
    )


def test_parse_qa_grounded_lines() -> None:
    text = (
        '{"task": "qa_grounded", '
        '"system": "You are a civic assistant.", '
        '"input": "Context:\\nFoo\\n\\nQuestion: What is foo?", '
        '"output": "Foo is bar per Item 1."}\n'
        '{"task": "qa_grounded", '
        '"system": "Sys", "input": "In", "output": "Out"}'
    )
    out = parse_examples(
        text=text,
        task=TaskType.QA_GROUNDED,
        chunk_id="d1#0",
        provenance=_provenance(),
    )
    assert len(out) == 2
    assert all(ex.task == TaskType.QA_GROUNDED for ex in out)
    assert all("d1#0" in ex.source_chunk_ids for ex in out)


def test_parse_drops_invalid_lines() -> None:
    text = (
        "garbage line\n"
        '{"task": "qa_grounded", "system": "S", "input": "I", "output": "O"}\n'
        '{"task": "qa_grounded", "system": ""}\n'  # empty system → ValidationError
    )
    out = parse_examples(
        text=text,
        task=TaskType.QA_GROUNDED,
        chunk_id="d1#0",
        provenance=_provenance(),
    )
    assert len(out) == 1


def test_parse_extract_normalizes_object_output() -> None:
    text = '{"task": "extract", "system": "S", "input": "I", "output": {"applicant": "Acme"}}'
    out = parse_examples(
        text=text,
        task=TaskType.EXTRACT,
        chunk_id="d1#0",
        provenance=_provenance(),
    )
    assert len(out) == 1
    assert "Acme" in out[0].output


def test_parse_strips_code_fences() -> None:
    text = '```json\n{"task": "summarize", "system": "S", "input": "I", "output": "O"}\n```'
    out = parse_examples(
        text=text,
        task=TaskType.SUMMARIZE,
        chunk_id="d1#0",
        provenance=_provenance(),
    )
    assert len(out) == 1
