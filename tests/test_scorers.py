"""Unit tests for per-bench scorers."""

from __future__ import annotations

from civic_slm.eval.scorers import score_extraction, score_factuality, score_refusal
from civic_slm.schema import ExtractionExample, FactualityExample, RefusalExample


def test_factuality_rewards_overlap_and_citations() -> None:
    ex = FactualityExample(
        id="f1",
        question="What is the population?",
        context="The 2020 census recorded the city's population as 64,581.",
        gold_answer="64,581 per the 2020 census",
        gold_citations=["2020 census"],
    )
    res = score_factuality(
        ex, "The population is 64,581 per the 2020 census.", model_id="m", latency_ms=10.0
    )
    assert res.score > 0.5


def test_factuality_penalizes_wrong_answer() -> None:
    ex = FactualityExample(
        id="f1",
        question="Q?",
        context="C",
        gold_answer="The applicant is Acme Corp.",
        gold_citations=["Acme"],
    )
    res = score_factuality(ex, "Bananas elephants moonshine.", model_id="m", latency_ms=1.0)
    assert res.score < 0.3


def test_refusal_correct_when_model_declines() -> None:
    ex = RefusalExample(id="r1", question="Q?", context="C")
    res = score_refusal(
        ex, "I don't know based on the provided context.", model_id="m", latency_ms=1.0
    )
    assert res.score == 1.0


def test_refusal_wrong_when_model_confabulates() -> None:
    ex = RefusalExample(id="r1", question="Q?", context="C")
    res = score_refusal(ex, "The answer is 42.", model_id="m", latency_ms=1.0)
    assert res.score == 0.0


def test_extraction_full_match() -> None:
    ex = ExtractionExample(
        id="e1",
        document_text="...",
        gold_json={"applicant": "Acme", "amount": 1000},
        schema_name="staff_report",
    )
    pred = '```json\n{"applicant": "Acme", "amount": 1000}\n```'
    res = score_extraction(ex, pred, model_id="m", latency_ms=1.0)
    assert res.score == 1.0


def test_extraction_partial_match() -> None:
    ex = ExtractionExample(
        id="e1",
        document_text="...",
        gold_json={"applicant": "Acme", "amount": 1000},
        schema_name="staff_report",
    )
    res = score_extraction(ex, '{"applicant": "Acme", "amount": 999}', model_id="m", latency_ms=1.0)
    # 1 of 2 keys correct, no extra fields → standard token F1:
    # tp=1, fp=0, fn=1, prec=1.0, rec=0.5 → F1 ≈ 0.667. The previous double-
    # count (charging wrong-value as both FP and FN) returned ~0.5 and
    # under-rewarded partially-correct extractions.
    assert abs(res.score - (2 / 3)) < 1e-6


def test_extraction_extra_field_only_counts_as_fp() -> None:
    """A hallucinated extra key is FP, not also FN — the gold has no slot for it."""
    ex = ExtractionExample(
        id="e1",
        document_text="...",
        gold_json={"applicant": "Acme"},
        schema_name="staff_report",
    )
    res = score_extraction(
        ex,
        '{"applicant": "Acme", "phantom": "x"}',
        model_id="m",
        latency_ms=1.0,
    )
    # tp=1, fp=1 (phantom), fn=0 → prec=0.5, rec=1.0, f1=0.667.
    assert abs(res.score - (2 / 3)) < 1e-6


def test_extraction_loose_equality_int_vs_string() -> None:
    """`"100"` should match a gold `100`; the JSON-loose-but-correct case."""
    ex = ExtractionExample(
        id="e1",
        document_text="...",
        gold_json={"amount": 100},
        schema_name="staff_report",
    )
    res = score_extraction(ex, '{"amount": "100"}', model_id="m", latency_ms=1.0)
    assert res.score == 1.0


def test_extraction_bool_not_loose_equal_to_int() -> None:
    """`True == 1` in Python but we don't want True to satisfy a gold integer 1."""
    from civic_slm.eval.scorers import _eq_loose

    assert _eq_loose(1, 1) is True
    assert _eq_loose(True, 1) is False
    assert _eq_loose(None, 0) is False
    assert _eq_loose("100", 100) is True
    assert _eq_loose("  yes  ", "yes") is True
