"""Per-bench scoring functions. Pure (prediction, example) → EvalResult.

These are deliberately lightweight at v0:
  - factuality: citation exact-match + word-overlap proxy for semantic similarity.
    The plan calls for BGE reranker similarity; we keep the door open by
    threading optional `embed_fn` through, but the default works without
    sentence-transformers installed.
  - refusal: regex over canonical refusal phrases.
  - extraction: field-level F1 against gold JSON.
  - side_by_side: defers scoring to a Claude judge — see judge.py.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from civic_slm.jsonparse import extract_first
from civic_slm.schema import (
    EvalResult,
    ExtractionExample,
    FactualityExample,
    RefusalExample,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# --- factuality ---------------------------------------------------------------

_REFUSAL_PATTERNS = [
    r"\bI (don'?t|do not) (know|have)\b",
    r"\bcannot (find|determine|answer)\b",
    r"\bnot (mentioned|in|provided|available|stated)\b",
    r"\bno (information|answer) (in|provided|given)\b",
    r"\bunable to (answer|find|determine)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def _word_overlap(a: str, b: str) -> float:
    """Jaccard over lower-cased word sets — cheap stand-in for semantic similarity."""
    aw = set(re.findall(r"\w+", a.lower()))
    bw = set(re.findall(r"\w+", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def score_factuality(
    example: FactualityExample,
    prediction: str,
    *,
    model_id: str,
    latency_ms: float,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> EvalResult:
    sim = (similarity_fn or _word_overlap)(example.gold_answer, prediction)
    # If the example has no gold citations, we can't reward citation matching;
    # the score is entirely the semantic similarity. Earlier versions gave a
    # free 1.0 in that case, which inflated scores by ~0.5 on citation-less
    # examples. Treat them as similarity-only to preserve score semantics.
    if example.gold_citations:
        citations_hit = sum(
            1 for c in example.gold_citations if c.lower() in prediction.lower()
        ) / len(example.gold_citations)
        score = 0.5 * sim + 0.5 * citations_hit
        notes = f"sim={sim:.2f} citations_hit={citations_hit:.2f}"
    else:
        score = sim
        notes = f"sim={sim:.2f} citations=none"
    # `sim` and `citations_hit` are each bounded [0, 1] and the convex
    # combination of two such values is also in [0, 1] — no clamp needed.
    return EvalResult(
        model_id=model_id,
        bench="factuality",
        example_id=example.id,
        prediction=prediction,
        score=score,
        judge_notes=notes,
        latency_ms=latency_ms,
    )


# --- refusal ------------------------------------------------------------------


def score_refusal(
    example: RefusalExample,
    prediction: str,
    *,
    model_id: str,
    latency_ms: float,
) -> EvalResult:
    refused = bool(_REFUSAL_RE.search(prediction))
    correct = refused == example.expected_refusal
    return EvalResult(
        model_id=model_id,
        bench="refusal",
        example_id=example.id,
        prediction=prediction,
        score=1.0 if correct else 0.0,
        judge_notes=f"refused={refused} expected={example.expected_refusal}",
        latency_ms=latency_ms,
    )


# --- extraction ---------------------------------------------------------------


def _extract_json(text: str) -> tuple[dict[str, object], str]:
    """Pull the first balanced JSON object out of `text`. Tolerant of code fences.

    Delegates to `civic_slm.jsonparse.extract_first` so the bracket-balanced
    scanner is shared with the browser-agent parser and synth reader; one
    correctness fix, three call sites benefit. See `jsonparse.Status` for
    the status vocabulary.
    """
    obj, status = extract_first(text, "object")
    if not isinstance(obj, dict):
        return {}, status
    return obj, status


def _eq_loose(a: object, b: object) -> bool:
    """Compare two extraction values, normalizing common type variations.

    Models routinely emit `"100"` for an integer 100, or `"2024-01-01"` for a
    date string the gold author wrote without quotes. Treating those as
    misses inflates the false-negative rate against models that are
    semantically correct but JSON-loose. We coerce both sides to a string and
    strip whitespace for comparison. This is more permissive than strict
    Python equality and matches what a human grader would do — and the
    extraction bench is small enough (n=50) that we trade tightness for
    fairness.
    """
    if a == b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):  # avoid 1 == True
        return False
    return str(a).strip() == str(b).strip()


def score_extraction(
    example: ExtractionExample,
    prediction: str,
    *,
    model_id: str,
    latency_ms: float,
) -> EvalResult:
    pred, status = _extract_json(prediction)
    gold = example.gold_json
    if not gold:
        f1 = 0.0
    else:
        tp = sum(1 for k, v in gold.items() if _eq_loose(pred.get(k), v))
        fp = sum(1 for k in pred if k not in gold or not _eq_loose(pred[k], gold.get(k)))
        fn = sum(1 for k in gold if not _eq_loose(pred.get(k), gold[k]))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    notes = f"parse={status} fields_gold={len(gold)} fields_pred={len(pred)}"
    if status != "ok":
        # Include a snippet of the raw prediction so you can see *why* parsing failed.
        notes += f" raw={prediction[:120]!r}"
    return EvalResult(
        model_id=model_id,
        bench="extraction",
        example_id=example.id,
        prediction=prediction,
        score=f1,
        judge_notes=notes,
        latency_ms=latency_ms,
    )
