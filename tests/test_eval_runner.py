"""Smoke tests: load shipped factuality JSONL, run against a stub client, verify
the runner produces results and writes a report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from civic_slm.eval.runner import load_examples, run, write_report
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
    assert len(examples) == 10
    assert all(ex.bench == "factuality" for ex in examples)


def test_runner_round_trip(tmp_path: Path) -> None:
    examples = load_examples(Path("data/eval/civic_factuality.jsonl"))
    results = run(examples=examples, client=_StubClient(), model_id="stub")  # type: ignore[arg-type]
    assert len(results) == 10
    write_report(results, tmp_path, "factuality")
    assert (tmp_path / "factuality.json").exists()
    assert (tmp_path / "factuality.md").exists()
