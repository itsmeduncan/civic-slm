# Design — Generative curation pass over the synth corpus (#57)

**Status:** approved (2026-07-23) · **Milestone:** v1.1.x · **Scope:** one Python subsystem + a `review-sft` upgrade

## Overview

Add `civic-slm curate {slug}` — a cheap (Haiku) first-pass curator that scores and
defect-classifies each synthetic SFT example, auto-splits the corpus into
accept / human-queue / reject, and feeds the queue (annotated with the predicted
defect + a suggested fix) into an upgraded `review-sft --queue`. Replaces the
"eyeball every example" flow for corpora too large to hand-review, and surfaces
_systemic_ defects (via a defect distribution) instead of one-offs.

## Goals / Non-goals

**Goals**

- Classify each example against a fixed defect taxonomy and score it 0–10.
- Auto-accept the clearly-good, auto-reject the training-poisoning, queue the rest.
- Annotate queued examples with the predicted defect + a suggested fix so the
  human's review is faster and targeted.
- Emit a defect distribution (which defects dominate → which prompt templates to fix).

**Non-goals (YAGNI / issue non-goals)**

- Web UI (terminal only).
- Auto-_applying_ suggested fixes (suggest-only; a human accepts or edits externally).
- Active-learning loop that retrains the curator from human decisions.
- Multi-rater agreement / inter-annotator stats.

## Defect taxonomy (the issue's hypothesis — refine after real runs)

`ungrounded_answer`, `schema_drift`, `leading_question`, `template_echo`,
`confused_refusal`, `pii_leak`, `wrong_jurisdiction_vocab`, `format_drift`.

**High-severity (force auto-reject):** `pii_leak` (privacy), `ungrounded_answer`
(teaches hallucination), `confused_refusal` (breaks refusal calibration). The
other five are _fixable_ → queue with a suggested fix.

## Architecture

| #   | File                                           | Change                                                                                                                                                                         |
| --- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `src/civic_slm/schema.py`                      | Add `DefectClass` enum, `HIGH_SEVERITY` frozenset, `CurationVerdict`, `QueuedExample`.                                                                                         |
| 2   | `src/civic_slm/synth/prompts/curate.txt` (new) | Curation prompt: taxonomy + 0–10 rubric + JSON output contract.                                                                                                                |
| 3   | `src/civic_slm/synth/curate.py` (new)          | `curate_example()` (one Haiku call → `CurationVerdict`), `disposition()` (verdict → bucket), `curate_corpus()` (async, concurrency-capped, resumable → 3-way split + summary). |
| 4   | `src/civic_slm/cli.py`                         | Register `app.command("curate")(curate_main)`.                                                                                                                                 |
| 5   | `src/civic_slm/synth/review.py`                | Add `--queue` mode: read `QueuedExample`, render the verdict annotation.                                                                                                       |
| 6   | `tests/test_curate.py` (new)                   | disposition table, verdict parsing, split writer + summary (backend mocked).                                                                                                   |

## Schema (schema.py)

```python
class DefectClass(StrEnum):
    UNGROUNDED_ANSWER = "ungrounded_answer"
    SCHEMA_DRIFT = "schema_drift"
    LEADING_QUESTION = "leading_question"
    TEMPLATE_ECHO = "template_echo"
    CONFUSED_REFUSAL = "confused_refusal"
    PII_LEAK = "pii_leak"
    WRONG_JURISDICTION_VOCAB = "wrong_jurisdiction_vocab"
    FORMAT_DRIFT = "format_drift"

HIGH_SEVERITY: frozenset[DefectClass] = frozenset(
    {DefectClass.PII_LEAK, DefectClass.UNGROUNDED_ANSWER, DefectClass.CONFUSED_REFUSAL}
)

class CurationVerdict(_Frozen):
    example_id: str
    score: int = Field(ge=0, le=10)
    defects: list[DefectClass] = Field(default_factory=list)
    suggested_fix: str | None = None
    rationale: str = Field(min_length=1)

class QueuedExample(_Frozen):   # one line of .curate-queue.jsonl / .rejected.jsonl
    example: InstructionExample
    verdict: CurationVerdict
```

## Disposition (pure function, unit-tested)

```python
class Bucket(StrEnum): ACCEPT = "accept"; QUEUE = "queue"; REJECT = "reject"

def disposition(v: CurationVerdict) -> Bucket:
    if any(d in HIGH_SEVERITY for d in v.defects) or v.score <= 3:
        return Bucket.REJECT
    if v.score >= 8 and not v.defects:
        return Bucket.ACCEPT
    return Bucket.QUEUE
```

Rationale: auto-accept is deliberately conservative — **any** defect (even a
"fixable" one) routes to the human so the suggested fix is seen; a false auto-accept
means training on bad data, a false queue only costs review time.

## Data flow

```
data/sft/{slug}.jsonl
  → curate_corpus() : for each example, Haiku → CurationVerdict → disposition()
      .curated.jsonl        (Bucket.ACCEPT — bare InstructionExample, ready for training)
      .curate-queue.jsonl   (Bucket.QUEUE — QueuedExample: example + verdict)
      .rejected.jsonl       (Bucket.REJECT — QueuedExample, for audit / template-debugging)
  → prints summary: {accept, queue, reject} counts + defect-class histogram
  → review-sft {slug} --queue : human triages the queue, accepts append to .curated.jsonl
```

**Verdict parsing:** the backend returns free text; parse the JSON object with
`civic_slm.jsonparse.extract_first(text, "object")`, then `CurationVerdict.model_validate`.

**Fail-safe:** a malformed/unparseable verdict, a validation error, or an
exhausted-retry API failure routes the example to **QUEUE** (with a synthetic
`rationale` noting the failure) — never silently dropped, never auto-accepted.

## CLI (cli.py → synth/curate.py `main`)

`civic-slm curate {slug}` with options mirroring `review-sft` + `synth`:
`--input` (default `data/sft/{slug}.jsonl`), `--out-dir` (default `data/sft/`),
`--model` (default `claude-haiku-4-5`, env `CIVIC_SLM_CURATOR_MODEL`),
`--concurrency` (default 8), `--limit`, `--data-dir`.
Backend via `select_backend(default_anthropic_model="claude-haiku-4-5")` — respects
`CIVIC_SLM_LLM_BACKEND` (anthropic|local) and the strict-local tripwire.
**Resumable:** a `{slug}.curate-state.json` records processed example ids (mirrors
`review.py`'s seen-set); a re-run skips done examples and appends. Async with a
`concurrency`-bounded semaphore (mirrors `synth/generate.py`).

## review-sft `--queue`

`civic-slm review-sft {slug} --queue [--queue-file …]` reads `.curate-queue.jsonl`
(`QueuedExample` lines). For each: render the example (as today) plus a rich panel
showing `verdict.score`, `verdict.defects`, `verdict.suggested_fix`, `verdict.rationale`.
Keys unchanged: `[a]ccept / [r]eject / [s]kip / [q]uit`; accepts append the bare
`InstructionExample` to `.curated.jsonl`. Suggest-only — no auto-edit. Without
`--queue`, `review-sft` behaves exactly as today (plain `.jsonl` of `InstructionExample`).

## Error handling

- Unparseable/invalid verdict → QUEUE (fail-safe), logged `warning`.
- API error → retry with backoff (reuse the eval-runner retry posture); on
  exhaustion → QUEUE + log, don't abort the whole run.
- Resumable state means a Ctrl-C or crash loses at most in-flight examples.
- `--input` missing / empty → actionable `typer.BadParameter`.

## Testing (`tests/test_curate.py`, backend mocked — MLX-skip-free)

- `disposition()` boundary table: score 3/4/7/8/9 × {no defect, each high-severity
  defect, a fixable defect}; assert the ACCEPT-needs-empty-defects rule and the
  high-severity-overrides-score rule.
- Verdict parsing: clean JSON; JSON-wrapped-in-prose; malformed → `None` → QUEUE.
- `curate_corpus()` with a stub backend over ~5 fixture examples → asserts the three
  output files' contents + the summary counts + defect histogram.
- `review-sft --queue`: loads a `QueuedExample` file and (via the existing
  input-injection test seam) renders without error; accept appends to `.curated.jsonl`.

## Out of scope / future

Auto-applying fixes; active-learning retrain of the curator; multi-rater stats; a
web UI. Taxonomy is the issue's hypothesis — the emitted defect histogram is the
empirical input for refining it and the synth prompt templates.
