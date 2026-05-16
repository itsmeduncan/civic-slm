"""Bracket-balanced JSON extraction for noisy model output.

Why this exists: model responses routinely wrap JSON in prose, code fences,
explanatory preambles, or trailing commentary. A naive "find first `{`, last
`}`" heuristic captures everything between them — including any intervening
prose — and silently corrupts the parse. We need a scanner that finds the
first *balanced* object or array and stops there, ignoring brackets inside
strings.

Three call sites need this: the extraction scorer (`eval/scorers.py`), the
browser-agent result parser (`ingest/recipes/_browser.py`), and the synth
generator's batched-object reader (`synth/generate.py`). Keeping one
implementation eliminates the "we fixed the bug in one place" failure mode.

`extract_first(text, kind)` returns `(parsed, status)`. `status` is one of:
  - ``ok``       — found and parsed.
  - ``no_open``  — no opening bracket of the requested kind.
  - ``unbalanced`` — opened but never closed at depth 0.
  - ``invalid_json`` — balanced but didn't ``json.loads``.
  - ``wrong_type`` — parsed but wrong top-level type for the kind.
"""

from __future__ import annotations

import json
import re
from typing import Literal

Kind = Literal["object", "array"]
Status = Literal["ok", "no_open", "unbalanced", "invalid_json", "wrong_type"]

_OPEN_CLOSE: dict[Kind, tuple[str, str]] = {
    "object": ("{", "}"),
    "array": ("[", "]"),
}

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove fenced-code markers without disturbing in-string backticks.

    We only strip ```` ``` ```` and ```` ```json ```` markers; we do not try
    to be clever about partial fences. Good enough for the prose-wrapped JSON
    models actually emit.
    """
    return _FENCE_RE.sub("", text)


def _find_balanced(text: str, open_ch: str, close_ch: str) -> tuple[int, int] | None:
    """Return the (start, end_exclusive) span of the first balanced bracket pair.

    Honors JSON string literals — a `}` inside `"..."` does NOT close the
    object. Backslash-escapes inside strings are respected so `"\"}"` is
    one character of content, not a string-end.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return (start, i + 1)
    return None


def extract_first(text: str, kind: Kind) -> tuple[object, Status]:
    """Pull the first balanced JSON object or array out of `text`.

    Returns `({}, status)` for objects and `([], status)` for arrays when
    parsing fails. Callers should surface the status string in logs or
    judge notes so failures stay diagnosable.
    """
    empty: object = {} if kind == "object" else []
    open_ch, close_ch = _OPEN_CLOSE[kind]
    cleaned = _strip_fences(text)
    if open_ch not in cleaned:
        return empty, "no_open"
    span = _find_balanced(cleaned, open_ch, close_ch)
    if span is None:
        return empty, "unbalanced"
    blob = cleaned[span[0] : span[1]]
    try:
        loaded = json.loads(blob)
    except (ValueError, json.JSONDecodeError):
        return empty, "invalid_json"
    if kind == "object" and not isinstance(loaded, dict):
        return empty, "wrong_type"
    if kind == "array" and not isinstance(loaded, list):
        return empty, "wrong_type"
    return loaded, "ok"
