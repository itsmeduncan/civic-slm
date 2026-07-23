# Synth-Corpus Curator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Haiku-driven `civic-slm curate {slug}` that scores + defect-classifies each synthetic SFT example, auto-splits accept/queue/reject, and an upgraded `review-sft --queue` that shows the predicted defect + suggested fix.

**Architecture:** New `synth/curate.py` (pure `disposition()` + `parse_verdict()` + async `curate_corpus()`), new pydantic contracts in `schema.py`, a prompt in `synth/prompts/curate.txt`, a `curate` CLI command, and a `--queue` mode on the existing `synth/review.py`. Mirrors the existing async/backend pattern in `synth/generate.py` and the resumable-state pattern in `synth/review.py`.

**Tech Stack:** Python 3.11, pydantic v2 (`_Frozen` base), `civic_slm.llm.backend` (Anthropic/local `complete()`), `civic_slm.jsonparse.extract_first`, Typer + rich, pytest.

## Global Constraints

- **Defect taxonomy (8):** `ungrounded_answer, schema_drift, leading_question, template_echo, confused_refusal, pii_leak, wrong_jurisdiction_vocab, format_drift`.
- **HIGH_SEVERITY (auto-reject):** `{pii_leak, ungrounded_answer, confused_refusal}`.
- **Disposition:** REJECT if any HIGH_SEVERITY defect **or** `score <= 3`; ACCEPT if `score >= 8` **and** `defects == []`; else QUEUE.
- **Fail-safe:** an unparseable/invalid verdict or an exhausted-retry API error routes the example to **QUEUE** (never dropped, never auto-accepted).
- **Curator model:** default `claude-haiku-4-5`, env override `CIVIC_SLM_CURATOR_MODEL`; obtained via `select_backend(default_anthropic_model="claude-haiku-4-5")` (respects `CIVIC_SLM_LLM_BACKEND` + strict-local).
- **Output files** (in `--out-dir`, default `data/sft/`): `{stem}.curated.jsonl` (bare `InstructionExample`), `{stem}.curate-queue.jsonl` and `{stem}.rejected.jsonl` (`QueuedExample`).
- **Suggest-only:** the curator never edits examples; `review-sft` displays the suggestion.
- **Style:** `from __future__ import annotations`, type hints on public fns, `_Frozen`/`StrEnum` per existing `schema.py`; ruff-clean; tests are MLX-skip-free and mock the backend.
- **Parse:** `obj, status = extract_first(text, "object")`; success is `status == "ok"`.

---

### Task 1: Schema contracts + `disposition()`

**Files:**

- Modify: `src/civic_slm/schema.py` (append `DefectClass`, `HIGH_SEVERITY`, `CurationVerdict`, `QueuedExample`)
- Create: `src/civic_slm/synth/curate.py` (`Bucket`, `disposition`)
- Create: `tests/test_curate.py`

**Interfaces:**

- Produces: `DefectClass(StrEnum)`, `HIGH_SEVERITY: frozenset[DefectClass]`, `CurationVerdict(_Frozen)` with fields `example_id:str, score:int(0-10), defects:list[DefectClass], suggested_fix:str|None, rationale:str`, `QueuedExample(_Frozen)` with `example:InstructionExample, verdict:CurationVerdict`.
- Produces: `Bucket(StrEnum) {ACCEPT,QUEUE,REJECT}`, `disposition(v: CurationVerdict) -> Bucket`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curate.py
from __future__ import annotations
import pytest
from civic_slm.schema import CurationVerdict, DefectClass
from civic_slm.synth.curate import Bucket, disposition

def _v(score: int, defects: list[DefectClass] | None = None) -> CurationVerdict:
    return CurationVerdict(example_id="e1", score=score, defects=defects or [], rationale="r")

@pytest.mark.parametrize("score,defects,expected", [
    (10, [], Bucket.ACCEPT),
    (8,  [], Bucket.ACCEPT),
    (9,  [DefectClass.FORMAT_DRIFT], Bucket.QUEUE),      # any defect blocks auto-accept
    (7,  [], Bucket.QUEUE),
    (4,  [], Bucket.QUEUE),
    (3,  [], Bucket.REJECT),
    (0,  [], Bucket.REJECT),
    (10, [DefectClass.PII_LEAK], Bucket.REJECT),          # high-severity overrides score
    (10, [DefectClass.UNGROUNDED_ANSWER], Bucket.REJECT),
    (10, [DefectClass.CONFUSED_REFUSAL], Bucket.REJECT),
    (6,  [DefectClass.SCHEMA_DRIFT], Bucket.QUEUE),       # fixable defect -> queue
])
def test_disposition(score, defects, expected):
    assert disposition(_v(score, defects)) is expected

def test_verdict_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        CurationVerdict(example_id="e", score=11, rationale="r")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_curate.py -v`
Expected: FAIL — `ModuleNotFoundError: civic_slm.synth.curate` / `ImportError: CurationVerdict`.

- [ ] **Step 3: Add schema contracts**

Append to `src/civic_slm/schema.py` (it already imports `StrEnum`, `Field`, `_Frozen`, and defines `InstructionExample`):

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
    """A curator's judgment of one InstructionExample."""

    example_id: str = Field(min_length=1)
    score: int = Field(ge=0, le=10)
    defects: list[DefectClass] = Field(default_factory=list)
    suggested_fix: str | None = None
    rationale: str = Field(min_length=1)


class QueuedExample(_Frozen):
    """One line of a curate-queue / rejected jsonl: the example plus its verdict."""

    example: InstructionExample
    verdict: CurationVerdict
```

- [ ] **Step 4: Create `curate.py` with `Bucket` + `disposition`**

```python
# src/civic_slm/synth/curate.py
"""`civic-slm curate` — model-driven first pass over a synth SFT corpus.

Scores + defect-classifies each InstructionExample (cheap Haiku call), then
routes it to accept / human-queue / reject. See
docs/superpowers/specs/2026-07-23-synth-curator-design.md.
"""
from __future__ import annotations

from enum import StrEnum

from civic_slm.schema import HIGH_SEVERITY, CurationVerdict


class Bucket(StrEnum):
    ACCEPT = "accept"
    QUEUE = "queue"
    REJECT = "reject"


def disposition(v: CurationVerdict) -> Bucket:
    """Route a verdict to a bucket.

    Conservative on ACCEPT: any defect (even a fixable one) sends the example to
    the human queue so the suggested fix is seen. A false auto-accept trains on
    bad data; a false queue only costs review time.
    """
    if any(d in HIGH_SEVERITY for d in v.defects) or v.score <= 3:
        return Bucket.REJECT
    if v.score >= 8 and not v.defects:
        return Bucket.ACCEPT
    return Bucket.QUEUE
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v` → PASS. Then `uv run ruff check src/civic_slm/schema.py src/civic_slm/synth/curate.py tests/test_curate.py` → clean.

- [ ] **Step 6: Commit**

```bash
git add src/civic_slm/schema.py src/civic_slm/synth/curate.py tests/test_curate.py
git commit -m "feat(synth): curation schema + disposition (#57)"
```

---

### Task 2: Curation prompt + `parse_verdict` + `curate_example`

**Files:**

- Create: `src/civic_slm/synth/prompts/curate.txt`
- Modify: `src/civic_slm/synth/curate.py`
- Modify: `tests/test_curate.py`

**Interfaces:**

- Consumes: `Backend` (`civic_slm.llm.backend`, `async complete(*, system, user, max_tokens) -> str`), `extract_first` (`civic_slm.jsonparse`), `InstructionExample`, `CurationVerdict` (Task 1).
- Produces: `parse_verdict(text: str, example_id: str) -> CurationVerdict | None`; `async curate_example(ex: InstructionExample, backend: Backend) -> CurationVerdict` (never raises — on failure returns a fail-safe verdict that `disposition()` routes to QUEUE).

- [ ] **Step 1: Write the failing test**

````python
# add to tests/test_curate.py
import json
from dataclasses import dataclass
from civic_slm.synth.curate import parse_verdict, curate_example
from civic_slm.schema import InstructionExample, TaskType, Provenance

def _example() -> InstructionExample:
    return InstructionExample(
        id="ex-1", task=TaskType.QA_GROUNDED, system="sys",
        input="Context:\nItem 8A raises water rates.", output="Item 8A raises water rates.",
        source_chunk_ids=["c1"],
        provenance=Provenance(prompt_sha="a"*64, model="claude", generator="claude"),
    )

def test_parse_verdict_clean_json():
    raw = json.dumps({"score": 9, "defects": [], "suggested_fix": None, "rationale": "grounded"})
    v = parse_verdict(raw, "ex-1")
    assert v is not None and v.score == 9 and v.example_id == "ex-1" and v.defects == []

def test_parse_verdict_json_in_prose():
    raw = 'Here is my assessment:\n```json\n{"score": 5, "defects": ["leading_question"], "rationale": "leads"}\n```'
    v = parse_verdict(raw, "ex-1")
    assert v is not None and v.score == 5 and v.defects[0].value == "leading_question"

def test_parse_verdict_malformed_returns_none():
    assert parse_verdict("not json at all", "ex-1") is None

import pytest
@pytest.mark.asyncio
async def test_curate_example_failsafe_on_bad_backend():
    @dataclass
    class BadBackend:
        model: str = "claude-haiku-4-5"
        async def complete(self, *, system, user, max_tokens=4096) -> str:
            return "garbage, no json"
    v = await curate_example(_example(), BadBackend())
    from civic_slm.synth.curate import disposition, Bucket
    assert v.example_id == "ex-1" and disposition(v) is Bucket.QUEUE  # fail-safe -> queue
````

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_curate.py -k "parse_verdict or failsafe" -v`
Expected: FAIL — `ImportError: parse_verdict`.

- [ ] **Step 3: Write the prompt**

```text
# src/civic_slm/synth/prompts/curate.txt
You are a strict data-quality curator for a civic-document instruction-tuning
corpus. Judge ONE (task, input, output) example. Return ONLY a JSON object.

Defect classes (include every one that applies, else []):
- ungrounded_answer: the output claims something not present in the input context.
- schema_drift: an extraction output uses wrong/renamed/nested keys.
- leading_question: the input question presupposes a fact that may not exist.
- template_echo: the output just restates the input with light rephrasing.
- confused_refusal: a refusal task's output actually answers instead of declining.
- pii_leak: output contains a real person's name/address from the source.
- wrong_jurisdiction_vocab: uses another jurisdiction's term (e.g. "CUP" for a Texas "SUP").
- format_drift: wrong format (bullets where prose asked, or vice versa).

Score 0-10: 10 = flawless training example; 0 = actively harmful. Be conservative.

Return exactly this shape (no prose outside the JSON):
{"score": <int 0-10>, "defects": [<defect strings>], "suggested_fix": <string or null>, "rationale": <one sentence>}

TASK: {task}
INPUT:
{input}
OUTPUT:
{output}
```

- [ ] **Step 4: Implement `parse_verdict` + `curate_example`**

Add to `src/civic_slm/synth/curate.py` (add imports at top: `from pathlib import Path`, `from civic_slm.jsonparse import extract_first`, `from civic_slm.llm.backend import Backend`, `from civic_slm.logging import get_logger`, `from civic_slm.schema import CurationVerdict, DefectClass, InstructionExample`; `log = get_logger(__name__)`; `_PROMPT = (Path(__file__).parent / "prompts" / "curate.txt").read_text(encoding="utf-8")`):

```python
def parse_verdict(text: str, example_id: str) -> CurationVerdict | None:
    """Parse a curator's raw reply into a CurationVerdict, or None if unusable."""
    obj, status = extract_first(text, "object")
    if status != "ok" or not isinstance(obj, dict):
        return None
    try:
        return CurationVerdict.model_validate({**obj, "example_id": example_id})
    except ValueError:
        return None


async def curate_example(ex: InstructionExample, backend: Backend) -> CurationVerdict:
    """Curate one example. Never raises: any failure yields a QUEUE-bound verdict."""
    user = _PROMPT.format(task=ex.task.value, input=ex.input, output=ex.output)
    try:
        text = await backend.complete(system=None, user=user, max_tokens=512)
    except Exception as exc:  # network / SDK / provider error — fail safe to queue
        log.warning("curate_backend_error", example_id=ex.id, error=str(exc))
        text = ""
    verdict = parse_verdict(text, ex.id)
    if verdict is None:
        log.warning("curate_unparseable", example_id=ex.id)
        # score 5 + no defects -> disposition() routes to QUEUE (never dropped/accepted)
        return CurationVerdict(example_id=ex.id, score=5, defects=[], rationale="curator output unparseable; queued for human")
    return verdict
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v` → PASS (all). `uv run ruff check src/civic_slm/synth/curate.py tests/test_curate.py` → clean.
Note: the async tests need `pytest-asyncio` (already a dep — `synth/generate.py` tests use it; `asyncio_mode = auto` is set in pyproject). If a test errors with "async def not natively supported", confirm `pyproject.toml` has `asyncio_mode = "auto"` (it does) and that the file has no local override.

- [ ] **Step 6: Commit**

```bash
git add src/civic_slm/synth/prompts/curate.txt src/civic_slm/synth/curate.py tests/test_curate.py
git commit -m "feat(synth): curation prompt + per-example curate + fail-safe (#57)"
```

---

### Task 3: `curate_corpus` — async split, summary, resumable

**Files:**

- Modify: `src/civic_slm/synth/curate.py`
- Modify: `tests/test_curate.py`

**Interfaces:**

- Consumes: `curate_example`, `disposition`, `Bucket` (Tasks 1–2); `InstructionExample`, `QueuedExample`.
- Produces: `@dataclass CurateSummary { accept:int, queue:int, reject:int, defects: dict[str,int] }`; `async curate_corpus(*, input_path: Path, out_dir: Path, backend: Backend, concurrency: int = 8, limit: int | None = None) -> CurateSummary`. Writes `{stem}.curated.jsonl` / `.curate-queue.jsonl` / `.rejected.jsonl`; resumable via `{stem}.curate-state.json` (a `{"seen": [...]}` set of example ids).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_curate.py
from pathlib import Path
from dataclasses import dataclass as _dc
from civic_slm.synth.curate import curate_corpus, CurateSummary
from civic_slm.schema import QueuedExample

@_dc
class StubBackend:
    """Scores by a marker in the output text so we can drive each bucket."""
    model: str = "claude-haiku-4-5"
    async def complete(self, *, system, user, max_tokens=4096) -> str:
        if "PII" in user:   return '{"score": 10, "defects": ["pii_leak"], "rationale": "name"}'
        if "GOOD" in user:  return '{"score": 9, "defects": [], "rationale": "clean"}'
        return '{"score": 6, "defects": ["format_drift"], "suggested_fix": "use prose", "rationale": "bullets"}'

def _ex(i: str, out: str) -> InstructionExample:
    return InstructionExample(id=i, task=TaskType.QA_GROUNDED, system="s", input="c", output=out,
        source_chunk_ids=["c1"], provenance=Provenance(prompt_sha="a"*64, model="m", generator="claude"))

@pytest.mark.asyncio
async def test_curate_corpus_splits_and_summarizes(tmp_path: Path):
    inp = tmp_path / "san.jsonl"
    inp.write_text("\n".join(e.model_dump_json() for e in [
        _ex("g1", "GOOD"), _ex("p1", "PII"), _ex("q1", "meh")]) + "\n", encoding="utf-8")
    summary = await curate_corpus(input_path=inp, out_dir=tmp_path, backend=StubBackend(), concurrency=2)
    assert (summary.accept, summary.queue, summary.reject) == (1, 1, 1)
    assert summary.defects["pii_leak"] == 1 and summary.defects["format_drift"] == 1
    assert (tmp_path / "san.curated.jsonl").read_text().count("g1") == 1
    q = [QueuedExample.model_validate_json(l) for l in (tmp_path / "san.curate-queue.jsonl").read_text().splitlines() if l.strip()]
    assert q[0].example.id == "q1" and q[0].verdict.suggested_fix == "use prose"
    # resumable: a second run over the same input adds nothing new
    s2 = await curate_corpus(input_path=inp, out_dir=tmp_path, backend=StubBackend(), concurrency=2)
    assert (s2.accept, s2.queue, s2.reject) == (0, 0, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_curate.py -k curate_corpus -v`
Expected: FAIL — `ImportError: curate_corpus`.

- [ ] **Step 3: Implement `curate_corpus`**

Add to `src/civic_slm/synth/curate.py` (add imports: `import asyncio`, `import json`, `from collections import Counter`, `from dataclasses import dataclass`, `from civic_slm.schema import QueuedExample`):

```python
@dataclass
class CurateSummary:
    accept: int = 0
    queue: int = 0
    reject: int = 0
    defects: dict[str, int] | None = None


def _state_path(input_path: Path) -> Path:
    return input_path.with_suffix(".curate-state.json")


def _load_seen(input_path: Path) -> set[str]:
    p = _state_path(input_path)
    return set(json.loads(p.read_text()).get("seen", [])) if p.exists() else set()


async def curate_corpus(
    *,
    input_path: Path,
    out_dir: Path,
    backend: Backend,
    concurrency: int = 8,
    limit: int | None = None,
) -> CurateSummary:
    """Curate every example in input_path; append to the 3 split files. Resumable."""
    stem = input_path.stem
    curated_p = out_dir / f"{stem}.curated.jsonl"
    queue_p = out_dir / f"{stem}.curate-queue.jsonl"
    reject_p = out_dir / f"{stem}.rejected.jsonl"
    seen = _load_seen(input_path)

    examples = [
        InstructionExample.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    todo = [e for e in examples if e.id not in seen]
    if limit is not None:
        todo = todo[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def one(ex: InstructionExample) -> tuple[InstructionExample, CurationVerdict]:
        async with sem:
            return ex, await curate_example(ex, backend)

    results = await asyncio.gather(*(one(e) for e in todo))

    counts = Counter[str]()
    defect_hist = Counter[str]()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        curated_p.open("a", encoding="utf-8") as fa,
        queue_p.open("a", encoding="utf-8") as fq,
        reject_p.open("a", encoding="utf-8") as fr,
    ):
        for ex, verdict in results:
            for d in verdict.defects:
                defect_hist[d.value] += 1
            bucket = disposition(verdict)
            counts[bucket.value] += 1
            if bucket is Bucket.ACCEPT:
                fa.write(ex.model_dump_json() + "\n")
            else:
                target = fq if bucket is Bucket.QUEUE else fr
                target.write(QueuedExample(example=ex, verdict=verdict).model_dump_json() + "\n")
            seen.add(ex.id)

    _state_path(input_path).write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
    return CurateSummary(
        accept=counts["accept"], queue=counts["queue"], reject=counts["reject"],
        defects=dict(defect_hist),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v` → PASS. `uv run ruff check src/civic_slm/synth/curate.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/civic_slm/synth/curate.py tests/test_curate.py
git commit -m "feat(synth): curate_corpus split + summary + resumable state (#57)"
```

---

### Task 4: `civic-slm curate` CLI command

**Files:**

- Modify: `src/civic_slm/synth/curate.py` (add `main`)
- Modify: `src/civic_slm/cli.py` (register)
- Modify: `tests/test_curate.py`

**Interfaces:**

- Consumes: `curate_corpus` (Task 3), `select_backend` (`civic_slm.llm.backend`), `settings` (`civic_slm.config`).
- Produces: `main(...)` Typer command, registered as `civic-slm curate`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_curate.py
from civic_slm.synth import curate as curate_mod

def test_cli_curate_runs(tmp_path: Path, monkeypatch):
    inp = tmp_path / "san.jsonl"
    inp.write_text(_ex("g1", "GOOD").model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(curate_mod, "select_backend", lambda **_: StubBackend())
    curate_mod.main(slug="san", input_path=inp, out_dir=tmp_path, model="claude-haiku-4-5",
                    concurrency=2, limit=None, data_dir=tmp_path)
    assert (tmp_path / "san.curated.jsonl").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_curate.py -k cli_curate -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'main'`.

- [ ] **Step 3: Implement `main`**

Add to `src/civic_slm/synth/curate.py` (add imports: `import typer`, `from rich.console import Console`, `from civic_slm.config import settings`, `from civic_slm.llm.backend import select_backend`, `from civic_slm.logging import configure`; `console = Console()`):

```python
def main(
    slug: str = typer.Argument(..., help="Jurisdiction slug (default input data/sft/{slug}.jsonl)."),
    input_path: Path | None = typer.Option(None, "--input", help="Override the synth jsonl to curate."),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Where the 3 split files land. Default: data/sft/."),
    model: str = typer.Option("claude-haiku-4-5", "--model", envvar="CIVIC_SLM_CURATOR_MODEL", help="Curator model."),
    concurrency: int = typer.Option(8, "--concurrency", min=1, help="Max concurrent curator calls."),
    limit: int | None = typer.Option(None, "--limit", help="Curate at most N examples this run."),
    data_dir: Path | None = typer.Option(None, "--data-dir", help="Override the project data dir."),
) -> None:
    """Score + defect-classify a synth SFT corpus; split into accept/queue/reject."""
    configure()
    base = data_dir or settings().data_dir
    inp = input_path or base / "sft" / f"{slug}.jsonl"
    if not inp.exists():
        raise typer.BadParameter(f"no synth corpus at {inp} — run `civic-slm synth {slug}` first, or pass --input.")
    out = out_dir or base / "sft"
    backend = select_backend(default_anthropic_model=model)
    summary = asyncio.run(
        curate_corpus(input_path=inp, out_dir=out, backend=backend, concurrency=concurrency, limit=limit)
    )
    console.print(
        f"[bold]curated {slug}[/bold]: "
        f"[green]{summary.accept} accept[/green] · "
        f"[yellow]{summary.queue} queue[/yellow] · [red]{summary.reject} reject[/red]"
    )
    if summary.defects:
        console.print("defects: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.defects.items())))
    console.print(f"→ review the queue: [cyan]civic-slm review-sft {slug} --queue[/cyan]")
```

Register in `src/civic_slm/cli.py` next to the other `app.command(...)` lines:

```python
from civic_slm.synth.curate import main as curate_main
app.command("curate")(curate_main)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v` → PASS. Then confirm registration: `uv run civic-slm curate --help` prints the options. `uv run ruff check src/civic_slm/synth/curate.py src/civic_slm/cli.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/civic_slm/synth/curate.py src/civic_slm/cli.py tests/test_curate.py
git commit -m "feat(synth): civic-slm curate CLI command (#57)"
```

---

### Task 5: `review-sft --queue`

**Files:**

- Modify: `src/civic_slm/synth/review.py`
- Modify: `tests/test_curate.py`

**Interfaces:**

- Consumes: `QueuedExample` (Task 1); the existing `review.py` `_load`, `_load_state`, `_save_state`, accept-append.
- Produces: a `--queue` boolean on `review.py:main`; a helper `_load_queue(path) -> list[QueuedExample]` and `_render_verdict(console, verdict) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_curate.py
from civic_slm.synth import review as review_mod

def test_review_loads_queue_file(tmp_path: Path):
    ex = _ex("q1", "meh")
    from civic_slm.schema import CurationVerdict as CV, DefectClass as DC
    qe = QueuedExample(example=ex, verdict=CV(example_id="q1", score=6, defects=[DC.FORMAT_DRIFT],
                       suggested_fix="use prose", rationale="bullets"))
    qp = tmp_path / "san.curate-queue.jsonl"
    qp.write_text(qe.model_dump_json() + "\n", encoding="utf-8")
    loaded = review_mod._load_queue(qp)
    assert len(loaded) == 1 and loaded[0].example.id == "q1" and loaded[0].verdict.suggested_fix == "use prose"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_curate.py -k review_loads_queue -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_load_queue'`.

- [ ] **Step 3: Implement `_load_queue` + `_render_verdict` + wire `--queue`**

Add to `src/civic_slm/synth/review.py` (add import `from civic_slm.schema import QueuedExample`; `from rich.panel import Panel`):

```python
def _load_queue(path: Path) -> list[QueuedExample]:
    return [
        QueuedExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render_verdict(console: Console, verdict) -> None:
    defects = ", ".join(d.value for d in verdict.defects) or "none"
    fix = verdict.suggested_fix or "—"
    console.print(
        Panel(
            f"[bold]score[/bold] {verdict.score}/10   [bold]defects[/bold] {defects}\n"
            f"[bold]why[/bold] {verdict.rationale}\n[bold]suggested fix[/bold] {fix}",
            title="curator", border_style="yellow",
        )
    )
```

Add a `queue: bool = typer.Option(False, "--queue", help="Review the curator's queue (.curate-queue.jsonl), showing each example's predicted defect + suggested fix.")` option to `main`. When `queue` is set: resolve `input_path` default to `{slug}.curate-queue.jsonl`, load with `_load_queue`, and in the per-example loop call `_render_verdict(console, qe.verdict)` before the accept/reject prompt, appending `qe.example` (the bare `InstructionExample`) to the curated output on accept. The non-queue path is unchanged. Reuse the existing `_load_state`/`_save_state` resumable seen-set keyed on `qe.example.id`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_curate.py -v` → PASS. Full sweep: `uv run pytest -q` → green. `uv run ruff check src/civic_slm/synth/review.py` and `uv run pyright src/civic_slm/synth/` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/civic_slm/synth/review.py tests/test_curate.py
git commit -m "feat(synth): review-sft --queue shows curator annotations (#57)"
```

---

## Self-Review

**Spec coverage:** DefectClass/HIGH_SEVERITY/verdict/QueuedExample (T1) ✓ · disposition rules (T1) ✓ · prompt + parse + fail-safe (T2) ✓ · 3-way split + summary + defect histogram + resumable (T3) ✓ · CLI + model/env/backend (T4) ✓ · review-sft --queue annotation, suggest-only (T5) ✓. Data-flow files (`.curated`/`.curate-queue`/`.rejected`) and their contents (bare vs QueuedExample) are asserted in T3. Fail-safe-to-QUEUE covered in T2 + T3's unparseable path.

**Placeholder scan:** every step has runnable code + exact commands; no TBD/"handle errors"-style gaps. The one prose step (T5 Step 3's `--queue` wiring) names the exact functions, option, default, and append target — no ambiguity.

**Type consistency:** `CurationVerdict(example_id, score, defects, suggested_fix, rationale)`, `QueuedExample(example, verdict)`, `disposition(v)->Bucket`, `parse_verdict(text, id)->CurationVerdict|None`, `curate_example(ex, backend)->CurationVerdict`, `curate_corpus(*, input_path, out_dir, backend, concurrency, limit)->CurateSummary` — names/signatures identical across T1–T5 and their tests.

**Known follow-up (out of scope):** taxonomy/prompt refinement from the emitted defect histogram after a real run; auto-applying suggested fixes; active-learning retrain.
