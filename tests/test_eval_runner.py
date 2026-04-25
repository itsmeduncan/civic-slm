"""Smoke tests: load shipped factuality JSONL, run against a stub client, verify
the runner produces results and writes a report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from civic_slm.eval.runner import (
    ContaminationError,
    assert_no_contamination,
    load_examples,
    run,
    write_report,
)
from civic_slm.ingest.manifest import append as manifest_append
from civic_slm.schema import CivicDocument, DocType, FactualityExample
from civic_slm.serve.client import ChatResponse


@dataclass
class _StubClient:
    """Returns the gold answer verbatim — confirms scorer wiring."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""

    def chat(self, system: str, user: str) -> ChatResponse:
        # We don't have access to the example here, so return a generic answer.
        # Tests assert on shape, not score.
        return ChatResponse(text="I don't know based on the provided context.", latency_ms=1.0)


def test_load_factuality_examples_validates() -> None:
    path = Path("data/eval/civic_factuality.jsonl")
    examples = load_examples(path)
    # The bench grows over time; the load+validate contract is what's under
    # test, not the exact count.
    assert len(examples) >= 10
    assert all(ex.bench == "factuality" for ex in examples)


def test_runner_round_trip(tmp_path: Path) -> None:
    examples = load_examples(Path("data/eval/civic_factuality.jsonl"))
    n = len(examples)
    results = run(examples=examples, client=_StubClient(), model_id="stub")  # type: ignore[arg-type]
    assert len(results) == n
    write_report(results, tmp_path, "factuality", run_config={"seed": 0, "temperature": 0.0})
    assert (tmp_path / "factuality.json").exists()
    assert (tmp_path / "factuality.md").exists()
    first_line = (tmp_path / "factuality.json").read_text().splitlines()[0]
    assert '"_run_config"' in first_line


def test_synthetic_examples_pass_contamination_check(tmp_path: Path) -> None:
    """Examples with `source_doc_hash is None` are treated as synthetic and pass."""
    examples = load_examples(Path("data/eval/civic_factuality.jsonl"))
    # No raw manifest, no source_doc_hash on any example — should not raise.
    assert_no_contamination(examples, data_dir=tmp_path)


def test_contamination_check_raises_on_overlap(tmp_path: Path) -> None:
    sha = "a" * 64
    doc = CivicDocument(
        id="ca/test/aaa",
        jurisdiction="test",
        state="CA",
        doc_type=DocType.AGENDA,
        source_url="https://example.gov/agenda.pdf",  # type: ignore[arg-type]
        retrieved_at=datetime.now(UTC),
        sha256=sha,
        raw_path="data/raw/aaa.pdf",
        text="hello world",
    )
    manifest_append(tmp_path, doc)
    poisoned = FactualityExample(
        id="bad",
        question="q",
        context="c",
        gold_answer="a",
        gold_citations=[],
        source_doc_hash=sha,
    )
    with pytest.raises(ContaminationError):
        assert_no_contamination([poisoned], data_dir=tmp_path)
    # Override flag converts the error into a logged warning, no raise.
    assert_no_contamination([poisoned], data_dir=tmp_path, allow_contamination=True)
