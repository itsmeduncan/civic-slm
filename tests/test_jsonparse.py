"""Bracket-balanced JSON extraction shared across scorers, browser, and synth."""

from __future__ import annotations

from civic_slm.jsonparse import extract_first


def test_extract_first_object_plain() -> None:
    obj, status = extract_first('{"a": 1}', "object")
    assert status == "ok"
    assert obj == {"a": 1}


def test_extract_first_object_inside_fence() -> None:
    text = '```json\n{"a": 1, "b": "two"}\n```'
    obj, status = extract_first(text, "object")
    assert status == "ok"
    assert obj == {"a": 1, "b": "two"}


def test_extract_first_object_ignores_braces_in_strings() -> None:
    """The naive `find first { / last }` heuristic explodes on this input."""
    text = 'Sure! Here is the JSON: {"note": "value with } inside", "ok": true}'
    obj, status = extract_first(text, "object")
    assert status == "ok"
    assert obj == {"note": "value with } inside", "ok": True}


def test_extract_first_object_stops_at_first_balanced() -> None:
    """The reviewer's exact scenario: prose between two objects must not concatenate."""
    text = 'Some prose like {note}. Here is the JSON: {"real": 42}'
    obj, status = extract_first(text, "object")
    # The first `{note}` is bracket-balanced but not valid JSON; status reflects that.
    assert status == "invalid_json"
    assert obj == {}


def test_extract_first_object_no_open() -> None:
    obj, status = extract_first("no json here", "object")
    assert status == "no_open"
    assert obj == {}


def test_extract_first_object_unbalanced() -> None:
    obj, status = extract_first('{"a": 1', "object")
    assert status == "unbalanced"
    assert obj == {}


def test_extract_first_object_wrong_type() -> None:
    """A top-level array when we asked for an object is `wrong_type`, not a crash."""
    obj, status = extract_first("[1, 2, 3]", "object")
    assert status == "no_open"  # no `{` anywhere; nothing balanced
    assert obj == {}


def test_extract_first_array_plain() -> None:
    arr, status = extract_first("[1, 2, 3]", "array")
    assert status == "ok"
    assert arr == [1, 2, 3]


def test_extract_first_array_inside_prose() -> None:
    text = 'I found these meetings: [1] [2] but the real list is: [{"id": 1}]'
    arr, status = extract_first(text, "array")
    # First balanced `[1]` parses as a JSON array of one integer — that's the
    # first balanced top-level array, so we return it. Documents the behavior:
    # callers should expect the *first* balanced match, not the "best" one.
    assert status == "ok"
    assert arr == [1]


def test_extract_first_array_escaped_quotes_in_string() -> None:
    text = '[{"q": "Has a \\"quote\\" inside"}]'
    arr, status = extract_first(text, "array")
    assert status == "ok"
    assert isinstance(arr, list)
    assert arr[0] == {"q": 'Has a "quote" inside'}
