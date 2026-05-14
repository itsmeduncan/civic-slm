"""Pin the synth `extract.md` prompt against the held-out eval bench.

#46 traced v1's extraction regression (-0.10 vs base) to a schema mismatch:
the synth prompt listed `agenda_item` / `contract` / `appeal_notice` /
`budget_item` while the eval bench keys on `meeting_agenda_item` /
`contract_award` / `public_hearing_notice` / `meeting_metadata`. The model
learned what the prompt taught and missed what the bench asked.

This test re-asserts the contract: every schema name + field list in the
prompt must match what `data/eval/structured_extraction.jsonl` actually
uses. A drift in either direction (prompt adds a schema the bench
doesn't have, bench adds a schema the prompt doesn't teach) fails here,
not at v_N+1 baseline time.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_PROMPT_PATH = Path("src/civic_slm/synth/prompts/extract.md")
_BENCH_PATH = Path("data/eval/structured_extraction.jsonl")


def _bench_schemas() -> dict[str, set[str]]:
    """Schema name → union of field names actually used in the bench."""
    out: dict[str, set[str]] = {}
    for line in _BENCH_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        name = rec["schema_name"]
        out.setdefault(name, set()).update(rec["gold_json"].keys())
    return out


def test_prompt_lists_every_bench_schema_name() -> None:
    """Every schema that appears in the bench must be named in the prompt
    so the teacher can produce examples for it."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    for schema_name in _bench_schemas():
        # Match the backticked form to avoid false positives on substrings.
        assert f"`{schema_name}`" in prompt, (
            f"schema {schema_name!r} is in the eval bench but missing from "
            f"the synth extract prompt — synth will never produce examples "
            f"for it. Add it to the schema list in extract.md."
        )


def test_prompt_field_names_match_bench() -> None:
    """For each schema, every field the bench uses must appear in the
    prompt's field list for that schema."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    for schema_name, fields in _bench_schemas().items():
        # Find the prompt line that names this schema. Format:
        # `- `schema_name`: `field_a`, `field_b`, ...`
        anchor = f"`{schema_name}`:"
        idx = prompt.find(anchor)
        assert idx != -1, f"schema {schema_name!r} listed but no fields"
        end = prompt.find("\n", idx)
        line = prompt[idx:end]
        for field in fields:
            assert f"`{field}`" in line, (
                f"schema {schema_name!r} uses field {field!r} in the bench, "
                f"but the prompt's line for {schema_name!r} doesn't mention it: "
                f"{line!r}"
            )


def test_prompt_warns_against_repeating_one_schema() -> None:
    """The lesson from v1: explicit instructions against single-schema
    outputs. If this language drops out of the prompt the regression risk
    returns."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    assert "vary the schema" in prompt.lower(), (
        "Prompt no longer instructs the teacher to vary schemas across "
        "examples; v1 regressed exactly because this instruction was absent."
    )


def test_prompt_does_not_list_unused_schemas() -> None:
    """If the prompt lists schemas the bench never uses, the teacher
    spends budget on examples that don't help the eval. Keep them in
    lockstep — adding a new schema requires extending the bench too."""
    prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    bench_names = set(_bench_schemas())
    # Schema lines start with `- `<name>``. Parse them out.
    prompt_schemas: set[str] = set()
    for raw in prompt.splitlines():
        line = raw.strip()
        if not line.startswith("- `"):
            continue
        # `- `<name>`: <fields...>`
        backtick = line.find("`", 3)
        if backtick == -1:
            continue
        prompt_schemas.add(line[3:backtick])
    extras = prompt_schemas - bench_names
    assert not extras, (
        f"Prompt lists schemas the eval bench doesn't use: {sorted(extras)}. "
        f"Either add them to data/eval/structured_extraction.jsonl or drop "
        f"them from the prompt."
    )


def test_bench_schema_distribution_is_documented() -> None:
    """Sanity that the v1-regression analysis still describes the bench
    shape — staff_report-heavy. If this drifts dramatically (e.g. an
    eval scale-up makes meeting_metadata modal), the prompt's
    'Prefer staff_report' rule needs updating."""
    counts = Counter(
        json.loads(line)["schema_name"]
        for line in _BENCH_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    modal_schema, modal_count = counts.most_common(1)[0]
    total = sum(counts.values())
    assert modal_schema == "staff_report", (
        f"Bench modal schema shifted to {modal_schema!r}; update extract.md's "
        f"'Prefer staff_report' rule (or its replacement)."
    )
    # staff_report should be at least 25% of the bench for the modal rule
    # to make sense; if it drops below that, revisit the rule.
    assert modal_count / total >= 0.25, (
        f"staff_report is {modal_count}/{total} = {modal_count / total:.0%}; "
        f"if it drops below 25% the 'Prefer staff_report' rule is wrong."
    )
