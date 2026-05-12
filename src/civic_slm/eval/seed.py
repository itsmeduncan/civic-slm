"""`civic-slm eval seed` — draft eval-bench candidates from real civic chunks.

Why a separate command instead of folding into `synth`: eval examples have a
**different schema** from SFT examples (held-out, per-bench JSON shapes in
`schema.py`). They also need a stricter quality bar — for factuality, every
gold citation must be a verbatim substring of the source chunk, or the
example is a contamination hazard.

Pipeline:
  1. Load processed chunks for the requested jurisdiction.
  2. For each chunk, prompt the configured backend (Anthropic or LM Studio
     via `select_backend()`) with a per-bench template.
  3. Parse + Pydantic-validate the JSONL response.
  4. Per-bench hard validation (verbatim-citation check for factuality;
     verbatim-context check for refusal / extraction).
  5. Append survivors to `data/eval/.staged-{bench}.jsonl` for the maintainer
     to review and promote into the canonical bench file.

All four benches (factuality / refusal / extraction / side_by_side) are wired.
Adding a fifth bench means adding a `.md` template, a `_validate_*` function,
and a row in `_VALIDATORS` / `_PROMPTS`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer

from civic_slm.config import settings
from civic_slm.ingest.processed import load_chunks
from civic_slm.llm.backend import select_backend
from civic_slm.logging import configure, get_logger
from civic_slm.schema import (
    EvalExample,
    ExtractionExample,
    FactualityExample,
    RefusalExample,
    SideBySideExample,
)

log = get_logger(__name__)

Bench = Literal["factuality", "refusal", "extraction", "side_by_side"]

_PROMPT_DIR = Path(__file__).parent / "prompts"
_PROMPTS: dict[Bench, str] = {
    "factuality": "factuality_seed.md",
    "refusal": "refusal_seed.md",
    "extraction": "extraction_seed.md",
    "side_by_side": "side_by_side_seed.md",
}

# Canonical bench-file paths (used by --promote and the user-facing reminder).
_CANONICAL: dict[Bench, str] = {
    "factuality": "civic_factuality.jsonl",
    "refusal": "refusal.jsonl",
    "extraction": "structured_extraction.jsonl",
    "side_by_side": "side_by_side.jsonl",
}


def _prompt_for(bench: Bench) -> str:
    return (_PROMPT_DIR / _PROMPTS[bench]).read_text(encoding="utf-8")


def _iter_json_objects(text: str):  # type: ignore[no-untyped-def]
    """Yield JSON objects from a model response.

    The prompt asks for JSONL but reasoning models sometimes emit a fenced
    code block, prose preamble, or comma-separated array. Try line-by-line
    first, then fall back to bracket-balanced object scanning.
    """
    cleaned = text.strip()
    # Strip surrounding code fences if present.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    # JSONL fast path.
    parsed = False
    for raw in cleaned.split("\n"):
        line = raw.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
            parsed = True
        except json.JSONDecodeError:
            continue
    if parsed:
        return
    # Bracket scan fallback (handles concatenated objects without newlines).
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blob = cleaned[start : i + 1]
                start = -1
                try:
                    yield json.loads(blob)
                except json.JSONDecodeError:
                    continue


# ---- per-bench validators ----------------------------------------------------


def _validate_factuality(
    candidate: dict[str, object],
    *,
    chunk_text: str,
    chunk_doc_hash: str | None,
    example_id: str,
) -> FactualityExample | None:
    """Reject candidates whose citations aren't verbatim in the chunk."""
    citations = candidate.get("gold_citations")
    if not isinstance(citations, list) or not citations:
        log.warning("seed_eval_dropped_no_citations", id=example_id)
        return None
    for cit in citations:
        if not isinstance(cit, str) or cit.strip() not in chunk_text:
            log.warning(
                "seed_eval_dropped_bad_citation",
                id=example_id,
                citation_preview=str(cit)[:80],
            )
            return None
    try:
        return FactualityExample.model_validate(
            {
                "id": example_id,
                "bench": "factuality",
                "question": candidate.get("question"),
                "context": candidate.get("context") or chunk_text,
                "gold_answer": candidate.get("gold_answer"),
                "gold_citations": citations,
                "source_doc_hash": chunk_doc_hash,
            }
        )
    except Exception as exc:
        log.warning("seed_eval_dropped_schema", id=example_id, error=str(exc))
        return None


def _validate_refusal(
    candidate: dict[str, object],
    *,
    chunk_text: str,
    chunk_doc_hash: str | None,
    example_id: str,
) -> RefusalExample | None:
    """Refusal examples must use the chunk verbatim as context."""
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        log.warning("seed_eval_dropped_no_question", id=example_id)
        return None
    try:
        return RefusalExample.model_validate(
            {
                "id": example_id,
                "bench": "refusal",
                "question": question,
                # Context is locked to the chunk so the bench is honest about
                # what's in the model's view — ignore whatever the model
                # echoed back so it can't subtly paraphrase.
                "context": chunk_text,
                "expected_refusal": bool(candidate.get("expected_refusal", True)),
                "source_doc_hash": chunk_doc_hash,
            }
        )
    except Exception as exc:
        log.warning("seed_eval_dropped_schema", id=example_id, error=str(exc))
        return None


def _validate_extraction(
    candidate: dict[str, object],
    *,
    chunk_text: str,
    chunk_doc_hash: str | None,
    example_id: str,
) -> ExtractionExample | None:
    """Extraction needs a flat gold_json and a schema_name."""
    schema_name = candidate.get("schema_name")
    gold_json = candidate.get("gold_json")
    if not isinstance(schema_name, str) or not schema_name.strip():
        log.warning("seed_eval_dropped_no_schema_name", id=example_id)
        return None
    if not isinstance(gold_json, dict) or not gold_json:
        log.warning("seed_eval_dropped_bad_gold_json", id=example_id)
        return None
    # Flatness check — drop nested-object schemas; current scorer is flat-F1.
    for v in gold_json.values():
        if isinstance(v, dict):
            log.warning(
                "seed_eval_dropped_nested_gold_json",
                id=example_id,
                schema_name=schema_name,
            )
            return None
    try:
        return ExtractionExample.model_validate(
            {
                "id": example_id,
                "bench": "extraction",
                "document_text": chunk_text,  # always the chunk verbatim
                "schema_name": schema_name,
                "gold_json": gold_json,
                "source_doc_hash": chunk_doc_hash,
            }
        )
    except Exception as exc:
        log.warning("seed_eval_dropped_schema", id=example_id, error=str(exc))
        return None


def _validate_side_by_side(
    candidate: dict[str, object],
    *,
    chunk_text: str,
    chunk_doc_hash: str | None,
    example_id: str,
) -> SideBySideExample | None:
    prompt = candidate.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        log.warning("seed_eval_dropped_no_prompt", id=example_id)
        return None
    try:
        return SideBySideExample.model_validate(
            {
                "id": example_id,
                "bench": "side_by_side",
                "prompt": prompt,
                "rubric": candidate.get("rubric"),
                "source_doc_hash": chunk_doc_hash,
            }
        )
    except Exception as exc:
        log.warning("seed_eval_dropped_schema", id=example_id, error=str(exc))
        return None


_Validator = Callable[..., EvalExample | None]
_VALIDATORS: dict[Bench, _Validator] = {
    "factuality": _validate_factuality,
    "refusal": _validate_refusal,
    "extraction": _validate_extraction,
    "side_by_side": _validate_side_by_side,
}


# ---- generation --------------------------------------------------------------


async def _generate_for_chunk(
    *,
    chunk_text: str,
    chunk_doc_hash: str | None,
    chunk_id: str,
    jurisdiction: str,
    state: str,
    doc_type: str,
    bench: Bench,
    n_per_chunk: int,
    max_tokens: int,
) -> list[EvalExample]:
    backend = select_backend()
    prompt = _prompt_for(bench).format(
        jurisdiction=jurisdiction,
        state=state,
        doc_type=doc_type,
        section_path="",
        chunk_text=chunk_text,
        n=n_per_chunk,
    )
    text = await backend.complete(system=None, user=prompt, max_tokens=max_tokens)
    validator = _VALIDATORS[bench]
    accepted: list[EvalExample] = []
    for i, candidate in enumerate(_iter_json_objects(text)):
        ex = validator(
            candidate,
            chunk_text=chunk_text,
            chunk_doc_hash=chunk_doc_hash,
            example_id=f"{bench}-{chunk_id}-{i:02d}",
        )
        if ex is not None:
            accepted.append(ex)
    return accepted


def main(
    jurisdiction: str = typer.Argument(
        ..., help="Jurisdiction slug whose processed chunks seed the bench."
    ),
    bench: Bench = typer.Option("factuality", "--bench", "-b", help="Which bench to draft for."),
    n_per_chunk: int = typer.Option(
        3, "--n-per-chunk", "-n", min=1, help="Candidate examples per chunk."
    ),
    max_tokens: int = typer.Option(
        8192,
        "--max-tokens",
        min=512,
        help=(
            "Backend max_tokens. Reasoning models (Qwen 3.6, Gemma 4) consume the first chunk "
            "as reasoning_content, so default is generous."
        ),
    ),
    out_path: Path | None = typer.Option(
        None,
        "--out",
        help="Output path. Default: data/eval/.staged-{bench}.jsonl (staging area for review).",
    ),
    promote: bool = typer.Option(
        False,
        "--promote",
        help=(
            "Append directly to the canonical bench file "
            "(data/eval/civic_factuality.jsonl etc.) instead of staging. "
            "Use only after a curation pass."
        ),
    ),
    data_dir: Path | None = typer.Option(
        None, help="Override data directory (default: <repo>/data)."
    ),
) -> None:
    """Draft eval-bench candidates from real civic chunks.

    Examples:
      civic-slm eval seed san-clemente
      civic-slm eval seed san-clemente -b refusal -n 2
      civic-slm eval seed san-clemente --promote   # only after curation
    """
    configure()
    target_dir = data_dir or settings().data_dir

    chunks = load_chunks(jurisdiction, data_dir=target_dir)
    if not chunks:
        raise typer.BadParameter(
            f"No processed chunks for {jurisdiction!r}. "
            f"Run `civic-slm process {jurisdiction}` first."
        )

    if out_path is None:
        if promote:
            out_path = target_dir / "eval" / _CANONICAL[bench]
        else:
            out_path = target_dir / "eval" / f".staged-{bench}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Seeding {bench} from {len(chunks)} chunk(s) x {n_per_chunk} candidates -> {out_path}"
    )

    async def run() -> list[EvalExample]:
        all_accepted: list[EvalExample] = []
        for c in chunks:
            chunk_id = f"{c.doc_id[:8]}-{c.chunk_idx:03d}"
            try:
                accepted = await _generate_for_chunk(
                    chunk_text=c.text,
                    chunk_doc_hash=c.source_doc_hash or c.doc_id,
                    chunk_id=chunk_id,
                    jurisdiction=jurisdiction,
                    state="--",
                    doc_type="--",
                    bench=bench,
                    n_per_chunk=n_per_chunk,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                log.warning(
                    "seed_eval_chunk_failed",
                    chunk=chunk_id,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                continue
            log.info(
                "seed_eval_chunk_done",
                chunk=chunk_id,
                accepted=len(accepted),
                requested=n_per_chunk,
            )
            all_accepted.extend(accepted)
        return all_accepted

    accepted = asyncio.run(run())

    with out_path.open("a", encoding="utf-8") as fh:
        for ex in accepted:
            fh.write(ex.model_dump_json() + "\n")

    typer.echo(
        f"Wrote {len(accepted)} {bench} examples "
        f"(of {len(chunks) * n_per_chunk} requested) -> {out_path}"
    )
    if not promote:
        typer.echo(
            "Review the staged file and merge into the canonical bench when ready:\n"
            f"  cat {out_path} >> {target_dir / 'eval' / _CANONICAL[bench]}"
        )


if __name__ == "__main__":
    typer.run(main)
