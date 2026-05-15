"""Smoke tests: load shipped factuality JSONL, run against a stub client, verify
the runner produces results and writes a report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from civic_slm.eval.runner import (
    ContaminationError,
    assert_no_contamination,
    load_examples,
    run,
    write_report,
)
from civic_slm.ingest.manifest import append as manifest_append
from civic_slm.schema import CivicDocument, DocType, FactualityExample
from civic_slm.serve.client import ChatClient, ChatResponse


@dataclass
class _StubClient:
    """Returns the gold answer verbatim — confirms scorer wiring."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""

    def chat(self, system: str, user: str) -> ChatResponse:
        # We don't have access to the example here, so return a generic answer.
        # Tests assert on shape, not score.
        return ChatResponse(text="I don't know based on the provided context.", latency_ms=1.0)


def test_load_factuality_examples_validates() -> None:
    path = Path("data/eval/civic_factuality.jsonl")
    examples = load_examples(path)
    # The bench grows over time; the load+validate contract is what's under
    # test, not the exact count.
    assert len(examples) >= 10
    assert all(ex.bench == "factuality" for ex in examples)


def test_runner_round_trip(tmp_path: Path) -> None:
    examples = load_examples(Path("data/eval/civic_factuality.jsonl"))
    n = len(examples)
    results = run(examples=examples, client=_StubClient(), model_id="stub")  # type: ignore[arg-type]
    assert len(results) == n
    write_report(results, tmp_path, "factuality", run_config={"seed": 0, "temperature": 0.0})
    assert (tmp_path / "factuality.json").exists()
    assert (tmp_path / "factuality.md").exists()
    first_line = (tmp_path / "factuality.json").read_text().splitlines()[0]
    assert '"_run_config"' in first_line


def test_synthetic_examples_pass_contamination_check(tmp_path: Path) -> None:
    """Examples with `source_doc_hash is None` are treated as synthetic and pass."""
    examples = load_examples(Path("data/eval/civic_factuality.jsonl"))
    # No raw manifest, no source_doc_hash on any example — should not raise.
    assert_no_contamination(examples, data_dir=tmp_path)


def _capture_payload(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch httpx.Client.post to capture and return synthetic 200 responses."""
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    real = httpx.Client

    def patched(**kwargs: object) -> httpx.Client:
        return real(transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", patched)
    return captured


def test_chat_template_kwargs_omitted_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = ChatClient(base_url="http://x", model="m")
    client.chat("sys", "hi")
    assert "chat_template_kwargs" not in captured[0]


def test_chat_template_kwargs_forwarded_to_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = ChatClient(
        base_url="http://x",
        model="m",
        chat_template_kwargs={"enable_thinking": False},
    )
    client.chat("sys", "hi")
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_contamination_check_raises_on_overlap(tmp_path: Path) -> None:
    sha = "a" * 64
    doc = CivicDocument(
        id="ca/test/aaa",
        jurisdiction="test",
        state="CA",
        doc_type=DocType.AGENDA,
        source_url="https://example.gov/agenda.pdf",  # type: ignore[arg-type]
        retrieved_at=datetime.now(UTC),
        sha256=sha,
        raw_path="data/raw/aaa.pdf",
        text="hello world",
    )
    manifest_append(tmp_path, doc)
    poisoned = FactualityExample(
        id="bad",
        question="q",
        context="c",
        gold_answer="a",
        gold_citations=[],
        source_doc_hash=sha,
    )
    with pytest.raises(ContaminationError):
        assert_no_contamination([poisoned], data_dir=tmp_path)
    # Override flag converts the error into a logged warning, no raise.
    assert_no_contamination([poisoned], data_dir=tmp_path, allow_contamination=True)


# ---------------------------------------------------------------------------
# Release-readiness gates (#55) — bench JSONL + artifact round-trip + the
# MODEL_CARD ↔ artifacts reconciliation.
# ---------------------------------------------------------------------------

import re  # noqa: E402
import subprocess  # noqa: E402

from pydantic import TypeAdapter  # noqa: E402

from civic_slm.schema import EvalResult  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_tracked(*roots: str) -> list[Path]:
    """Return repo-relative paths tracked by git under the given roots.

    Using `git ls-files` keeps tests off gitignored helpers (e.g.
    `data/eval/.jurisdiction-tags.jsonl`) without an explicit exclusion list.
    """
    proc = subprocess.run(
        ["git", "ls-files", *roots],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in proc.stdout.splitlines() if line.strip()]


def _tracked_bench_files() -> list[Path]:
    return [p for p in _git_tracked("data/eval") if p.suffix == ".jsonl"]


def _tracked_eval_artifacts() -> list[Path]:
    return [p for p in _git_tracked("artifacts/evals") if p.suffix == ".json"]


@pytest.mark.parametrize(
    "bench_file",
    _tracked_bench_files(),
    ids=lambda p: p.name,
)
def test_every_committed_bench_jsonl_validates(bench_file: Path) -> None:
    """Every line in `data/eval/*.jsonl` must round-trip through the
    discriminated `EvalExample` union. A hand-edit that breaks the schema
    (renamed field, wrong literal) would otherwise only surface at eval-run
    time. See #55.
    """
    examples = load_examples(bench_file)
    assert examples, f"{bench_file.name}: empty bench file"
    # Bench label matches the filename family.
    name_to_bench = {
        "civic_factuality.jsonl": "factuality",
        "refusal.jsonl": "refusal",
        "structured_extraction.jsonl": "extraction",
        "side_by_side.jsonl": "side_by_side",
    }
    expected = name_to_bench.get(bench_file.name)
    if expected is not None:
        assert all(ex.bench == expected for ex in examples), (
            f"{bench_file.name}: examples with non-{expected!r} bench tag"
        )


_RESULT_ADAPTER: TypeAdapter[EvalResult] = TypeAdapter(EvalResult)


@pytest.mark.parametrize(
    "artifact_path",
    _tracked_eval_artifacts(),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_every_committed_eval_artifact_round_trips(artifact_path: Path) -> None:
    """Every `artifacts/evals/<model>/<bench>.json` must parse with the
    current runner's schemas. The runner emits an optional `_run_config`
    sentinel on line 1; every subsequent line is an `EvalResult`. A
    mismatched column would mean MODEL_CARD numbers came from a stale
    runner version. See #55.
    """
    # `second-city-breakdown.json` is a hand-authored analysis artifact, not
    # a runner output — it doesn't follow the per-line EvalResult contract.
    if artifact_path.stem == "second-city-breakdown":
        pytest.skip("breakdown analysis, not a runner artifact")

    lines = [
        line for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert lines, f"{artifact_path}: empty"

    head = json.loads(lines[0])
    # Sentinel is optional — some legacy artifacts pre-date it. If absent,
    # line 1 must already be a valid result.
    result_lines = lines[1:] if "_run_config" in head else lines

    for i, line in enumerate(result_lines, start=2 if "_run_config" in head else 1):
        try:
            _RESULT_ADAPTER.validate_json(line)
        except Exception as exc:
            pytest.fail(f"{artifact_path}:{i}: failed EvalResult schema — {exc}")


# Bench name (as written in MODEL_CARD) → artifact filename stem.
_MODEL_CARD_BENCH_TO_ARTIFACT = {
    "civic_factuality": "factuality",
    "refusal": "refusal",
    "structured_extraction": "extraction",
    "side_by_side": "side_by_side",
}

# MODEL_CARD column header → artifacts/evals subdirectory.
_MODEL_CARD_COL_TO_DIR = {
    "Base Qwen 3.6 27B": "base-qwen3.6-27b",
    "v1 (san-clemente-v1)": "san-clemente-v1",
    "v1.1 (civic-slm-v11)": "civic-slm-v11",
}

# Cell values that mean "no committed artifact for this combination."
_NO_RESULT_MARKERS = {"n/a", "not run", "pending", "—", "-"}


def _parse_model_card_row(row: str) -> list[str]:
    """Split a markdown table row into stripped cell text."""
    # Drop leading/trailing `|` then split.
    inner = row.strip().strip("|")
    return [c.strip() for c in inner.split("|")]


def _strip_md_emphasis(cell: str) -> str:
    """Strip `**bold**`, backticks, and surrounding whitespace."""
    cell = cell.strip()
    cell = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cell)
    cell = cell.strip("` ")
    return cell


def _cell_to_score(cell: str) -> float | None:
    """Parse a numeric score cell or return None if the cell is a no-result
    marker / a target threshold (`≥ 0.65`)."""
    s = _strip_md_emphasis(cell).lower()
    if s in _NO_RESULT_MARKERS or s.startswith("≥") or s.startswith(">="):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def test_model_card_numbers_match_artifacts() -> None:
    """Every numeric score cell in the MODEL_CARD eval table must match the
    mean score computed from `artifacts/evals/<model>/<bench>.json`. Drift
    here means "I can't reproduce your numbers" — the v1 blocker the
    issue calls out. See #55.

    Allowed tolerance is 0.0005 — MODEL_CARD shows 4-decimal precision so
    legitimate values match exactly; the tolerance just absorbs trailing-
    rounding ambiguity (0.49524 → "0.4952" vs "0.4953").
    """
    text = (REPO_ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    # Find the eval table by its header row.
    header_re = re.compile(r"^\| Benchmark .*\| v1 target.*\|$", re.MULTILINE)
    m = header_re.search(text)
    assert m, "MODEL_CARD.md: eval table header not found — has the table moved?"

    header_cells = _parse_model_card_row(text[m.start() : text.find("\n", m.end())])
    score_columns = {
        i: name for i, name in enumerate(header_cells) if name in _MODEL_CARD_COL_TO_DIR
    }
    assert score_columns, f"MODEL_CARD.md: none of {list(_MODEL_CARD_COL_TO_DIR)} found in header"

    # Walk the body rows until we leave the table (blank line / non-`|` line).
    body_start = text.find("\n", m.end()) + 1
    # Skip the separator row `| --- | --- | ... |`.
    body_start = text.find("\n", body_start) + 1
    body = text[body_start:]
    rows = []
    for line in body.splitlines():
        if not line.startswith("|"):
            break
        rows.append(line)
    assert rows, "MODEL_CARD.md: no body rows found under header"

    mismatches: list[str] = []
    for row in rows:
        cells = _parse_model_card_row(row)
        bench_name = _strip_md_emphasis(cells[0])
        artifact_bench = _MODEL_CARD_BENCH_TO_ARTIFACT.get(bench_name)
        if artifact_bench is None:
            continue
        for col_idx, model_col in score_columns.items():
            if col_idx >= len(cells):
                continue
            claimed = _cell_to_score(cells[col_idx])
            if claimed is None:
                continue
            artifact_dir = _MODEL_CARD_COL_TO_DIR[model_col]
            artifact = REPO_ROOT / "artifacts" / "evals" / artifact_dir / f"{artifact_bench}.json"
            if not artifact.exists():
                mismatches.append(
                    f"{model_col} / {bench_name}: claims {claimed} but "
                    f"{artifact.relative_to(REPO_ROOT)} is missing"
                )
                continue
            results = [
                _RESULT_ADAPTER.validate_json(line)
                for line in artifact.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith('{"_run_config"')
            ]
            if not results:
                mismatches.append(f"{artifact.relative_to(REPO_ROOT)}: no result rows")
                continue
            actual = sum(r.score for r in results) / len(results)
            if abs(actual - claimed) > 5e-4:
                mismatches.append(
                    f"{model_col} / {bench_name}: MODEL_CARD={claimed:.4f} "
                    f"vs artifact mean={actual:.4f} "
                    f"({artifact.relative_to(REPO_ROOT)})"
                )

    assert not mismatches, "MODEL_CARD numbers drifted from artifacts/evals/:\n  " + "\n  ".join(
        mismatches
    )
