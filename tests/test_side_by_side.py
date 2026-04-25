"""Tests for the side_by_side runner.

The runner orchestrates two ChatClients + a Claude judge. We stub all three
so the test stays fast and offline. The 72B comparator is the operational
gate — we verify the no-comparator skip path raises a clear error instead
of crashing on the first chat call.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from civic_slm.eval.judge import JudgeVerdict
from civic_slm.eval.side_by_side import (
    ComparatorMissingError,
    _ping_comparator,  # pyright: ignore[reportPrivateUsage] — testing internal helper
    run_side_by_side,
)
from civic_slm.schema import SideBySideExample
from civic_slm.serve.client import ChatResponse


@dataclass
class _StubClient:
    label: str
    base_url: str = ""
    model: str = ""
    api_key: str = ""

    def chat(self, _system: str, _user: str) -> ChatResponse:
        return ChatResponse(text=f"answer-from-{self.label}", latency_ms=1.0)


def test_run_side_by_side_computes_winrate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the judge to always return 'A' (the candidate). Win-rate must be 1.0."""

    def _judge(
        *,
        prompt: str,
        rubric: str,
        response_a: str,
        response_b: str,
        model: str,
    ) -> JudgeVerdict:
        return JudgeVerdict(winner="A", reason="stubbed")

    monkeypatch.setattr("civic_slm.eval.side_by_side.judge_with_position_swap", _judge)
    examples = [
        SideBySideExample(id="s1", prompt="What is a CUP?", rubric="accuracy"),
        SideBySideExample(id="s2", prompt="Define general plan.", rubric="accuracy"),
    ]
    results = run_side_by_side(
        examples=examples,
        candidate=_StubClient("cand"),  # type: ignore[arg-type]
        comparator=_StubClient("comp"),  # type: ignore[arg-type]
        candidate_id="stub-candidate",
    )
    assert len(results) == 2
    assert all(r.score == 1.0 for r in results)


def test_run_side_by_side_handles_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    def _judge(
        *,
        prompt: str,
        rubric: str,
        response_a: str,
        response_b: str,
        model: str,
    ) -> JudgeVerdict:
        return JudgeVerdict(winner="tie", reason="indistinguishable")

    monkeypatch.setattr("civic_slm.eval.side_by_side.judge_with_position_swap", _judge)
    examples = [SideBySideExample(id="s1", prompt="x", rubric=None)]
    results = run_side_by_side(
        examples=examples,
        candidate=_StubClient("cand"),  # type: ignore[arg-type]
        comparator=_StubClient("comp"),  # type: ignore[arg-type]
        candidate_id="stub",
    )
    assert results[0].score == 0.5


def test_ping_comparator_raises_on_unreachable() -> None:
    """A bogus port must surface ComparatorMissingError, not a generic httpx error."""
    # Use an unallocated localhost port so the connection refuses immediately.
    with pytest.raises(ComparatorMissingError, match="not reachable"):
        _ping_comparator("http://127.0.0.1:1", "any", timeout_s=0.5)
