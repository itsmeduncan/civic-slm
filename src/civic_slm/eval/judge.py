"""Pairwise LLM judge for the side_by_side bench.

Why pairwise instead of absolute: civic answers are open-ended; absolute
quality scores are noisy and drift across rubrics. Pairwise asks "is A or B
better given this rubric" — which calibrates better and matches how human
reviewers actually compare model responses.

Position bias mitigation: every comparison is run twice with A/B swapped.
A model only "wins" if it wins both orderings; otherwise the result is a tie.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from civic_slm.config import require

JUDGE_MODEL_DEFAULT = "claude-sonnet-4-6"

_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating two AI responses to the same prompt about "
    "California municipal government. Compare them on accuracy, faithfulness to "
    "California law and practice, and clarity. Penalize fabrication of specific facts "
    "(dates, dollar amounts, code sections) that aren't grounded.\n\n"
    'Reply with strictly valid JSON: {"winner": "A" | "B" | "tie", "reason": "<one '
    'sentence>"}. No prose outside the JSON.'
)

_JUDGE_USER_TEMPLATE = """\
Prompt:
{prompt}

Rubric: {rubric}

Response A:
{a}

Response B:
{b}

Decide. JSON only."""


@dataclass(frozen=True)
class JudgeVerdict:
    winner: Literal["A", "B", "tie"]
    reason: str


def judge_pair(
    *,
    prompt: str,
    rubric: str,
    response_a: str,
    response_b: str,
    model: str = JUDGE_MODEL_DEFAULT,
) -> JudgeVerdict:
    """Single-call pairwise judgment. Caller should run twice with swapped order."""
    from anthropic import Anthropic  # type: ignore[import-not-found]  # optional dep

    client = Anthropic(api_key=require("ANTHROPIC_API_KEY"))
    msg = client.messages.create(  # pyright: ignore[reportUnknownMemberType]
        model=model,
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_USER_TEMPLATE.format(
                    prompt=prompt, rubric=rubric or "general quality", a=response_a, b=response_b
                ),
            }
        ],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    return parse_verdict(text)


def judge_with_position_swap(
    *,
    prompt: str,
    rubric: str,
    response_a: str,
    response_b: str,
    model: str = JUDGE_MODEL_DEFAULT,
) -> JudgeVerdict:
    """Run the judge twice with A/B swapped; agree to count, else 'tie'."""
    forward = judge_pair(
        prompt=prompt,
        rubric=rubric,
        response_a=response_a,
        response_b=response_b,
        model=model,
    )
    reverse = judge_pair(
        prompt=prompt,
        rubric=rubric,
        response_a=response_b,
        response_b=response_a,
        model=model,
    )
    # Reverse pass: A in second call corresponds to B in first; flip its winner.
    flipped: Literal["A", "B", "tie"] = (
        "B" if reverse.winner == "A" else "A" if reverse.winner == "B" else "tie"
    )
    if forward.winner == flipped and forward.winner != "tie":
        return JudgeVerdict(winner=forward.winner, reason=forward.reason)
    return JudgeVerdict(winner="tie", reason=f"split: forward={forward.winner} reverse={flipped}")


def parse_verdict(text: str) -> JudgeVerdict:
    cleaned = re.sub(r"```(json)?", "", text)
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        data = json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return JudgeVerdict(winner="tie", reason=f"unparseable judge output: {text[:200]}")
    winner = data.get("winner")
    if winner not in {"A", "B", "tie"}:
        return JudgeVerdict(winner="tie", reason="invalid winner field")
    return JudgeVerdict(winner=winner, reason=str(data.get("reason", "")))
