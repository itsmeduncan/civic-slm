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

import json
import re
from typing import TYPE_CHECKING

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
    return EvalResult(
        model_id=model_id,
        bench="factuality",
        example_id=example.id,
        prediction=prediction,
        score=max(0.0, min(1.0, score)),
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

    Returns `(parsed_dict, status)` where status is one of:
    `"ok"`, `"no_braces"` (model didn't emit a JSON object), `"invalid_json"`
    (emitted something that looks like JSON but doesn't parse), or
    `"not_object"` (parsed but it was an array/scalar). Callers should surface
    the status in `judge_notes` so failures are diagnosable from the report.
    """
    cleaned = re.sub(r"```(json)?", "", text)
    if "{" not in cleaned:
        return {}, "no_braces"
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        loaded = json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}, "invalid_json"
    if not isinstance(loaded, dict):
        return {}, "not_object"
    return loaded, "ok"


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
        tp = sum(1 for k, v in gold.items() if pred.get(k) == v)
        fp = sum(1 for k in pred if k not in gold or pred[k] != gold.get(k))
        fn = sum(1 for k in gold if pred.get(k) != gold[k])
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
