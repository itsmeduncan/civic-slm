"""Unit tests for the verdict parser. The Anthropic call itself is not exercised."""

from __future__ import annotations

from civic_slm.eval.judge import parse_verdict


def testparse_verdict_a() -> None:
    v = parse_verdict('{"winner": "A", "reason": "more accurate"}')
    assert v.winner == "A"
    assert "accurate" in v.reason


def testparse_verdict_with_code_fence() -> None:
    v = parse_verdict('```json\n{"winner": "B", "reason": "clearer"}\n```')
    assert v.winner == "B"


def testparse_verdict_invalid_winner_falls_to_tie() -> None:
    v = parse_verdict('{"winner": "yes", "reason": "x"}')
    assert v.winner == "tie"


def testparse_verdict_unparseable_falls_to_tie() -> None:
    v = parse_verdict("garbage no json here")
    assert v.winner == "tie"
