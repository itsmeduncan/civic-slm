"""Eval runner: loads a benchmark JSONL, asks the model, scores, writes a report.

CLI: `python -m civic_slm.eval.runner --model qwen-civic-base --bench factuality \\
        --base-url http://localhost:8080 --bench-file data/eval/civic_factuality.jsonl`
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import TypeAdapter

from civic_slm.config import settings
from civic_slm.eval.scorers import score_extraction, score_factuality, score_refusal
from civic_slm.ingest.manifest import known_hashes
from civic_slm.logging import configure, get_logger
from civic_slm.schema import (
    EvalExample,
    EvalResult,
    ExtractionExample,
    FactualityExample,
    RefusalExample,
)
from civic_slm.serve import runtimes
from civic_slm.serve.client import ChatClient


class ContaminationError(RuntimeError):
    """Raised when an eval example's source document is also in the train manifest."""


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

log = get_logger(__name__)
app = typer.Typer(help="Run an evaluation benchmark against a served model.")

EVAL_ADAPTER: TypeAdapter[EvalExample] = TypeAdapter(EvalExample)

_FACTUALITY_SYSTEM = (
    "You are a civic assistant. Answer the user's question using ONLY the provided "
    "context. Cite specific section names or item numbers from the context. If the "
    "answer is not in the context, say you don't know."
)
_EXTRACTION_SYSTEM = (
    "You are a civic data extractor. Given a document, return a JSON object with the "
    "requested fields. Output ONLY the JSON object, no prose."
)


def load_examples(path: Path) -> list[EvalExample]:
    return [EVAL_ADAPTER.validate_json(line) for line in _iter_lines(path)]


def assert_no_contamination(
    examples: list[EvalExample],
    *,
    data_dir: Path,
    allow_contamination: bool = False,
) -> None:
    """Raise `ContaminationError` if any eval example's source doc is in the train manifest.

    Examples with `source_doc_hash is None` are treated as synthetic and pass
    trivially; the check binds the moment a real document is referenced.

    Pass `allow_contamination=True` only with an explicit operator decision —
    we still log a loud warning so the run is auditable.
    """
    eval_hashes = {ex.source_doc_hash for ex in examples if ex.source_doc_hash}
    if not eval_hashes:
        return
    train_hashes = known_hashes(data_dir)
    overlap = eval_hashes & train_hashes
    if not overlap:
        return
    if allow_contamination:
        log.warning(
            "eval_contamination_overridden",
            overlap_count=len(overlap),
            sample=sorted(overlap)[:3],
        )
        return
    raise ContaminationError(
        f"{len(overlap)} eval example(s) share source documents with the train manifest "
        f"at {data_dir}/raw/manifest.jsonl. Sample hashes: {sorted(overlap)[:3]}. "
        "Re-run with --allow-contamination to override (not recommended)."
    )


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped


def run(
    *,
    examples: list[EvalExample],
    client: ChatClient,
    model_id: str,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> list[EvalResult]:
    """Run an evaluation. `similarity_fn` is forwarded to `score_factuality`.

    `None` keeps the legacy word-overlap behavior so pre-v0.2 baselines remain
    reproducible. Pass `bge_similarity_fn()` (from `civic_slm.eval.embeddings`)
    to use the BGE dual-encoder cosine.
    """
    results: list[EvalResult] = []
    for ex in examples:
        if isinstance(ex, FactualityExample):
            user = f"Context:\n{ex.context}\n\nQuestion: {ex.question}"
            resp = client.chat(_FACTUALITY_SYSTEM, user)
            results.append(
                score_factuality(
                    ex,
                    resp.text,
                    model_id=model_id,
                    latency_ms=resp.latency_ms,
                    similarity_fn=similarity_fn,
                )
            )
        elif isinstance(ex, RefusalExample):
            user = f"Context:\n{ex.context}\n\nQuestion: {ex.question}"
            resp = client.chat(_FACTUALITY_SYSTEM, user)
            results.append(
                score_refusal(ex, resp.text, model_id=model_id, latency_ms=resp.latency_ms)
            )
        elif isinstance(ex, ExtractionExample):
            user = (
                f"Schema: {ex.schema_name}\n\nDocument:\n{ex.document_text}\n\nReturn the JSON now."
            )
            resp = client.chat(_EXTRACTION_SYSTEM, user)
            results.append(
                score_extraction(ex, resp.text, model_id=model_id, latency_ms=resp.latency_ms)
            )
        else:
            # SideBySideExample requires a pairwise judge — separate runner (Phase 2).
            log.info("skipping_side_by_side", id=ex.id)
            continue
    return results


def write_report(
    results: list[EvalResult],
    out_dir: Path,
    bench: str,
    *,
    run_config: dict[str, object] | None = None,
) -> None:
    """Write JSONL of results plus a markdown summary.

    `run_config` is recorded as the first line of the JSONL (under a
    `_run_config` key) and at the top of the markdown — minimum: seed,
    temperature, max_tokens, served model name, base URL. This is what
    makes a baseline reproducible across runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{bench}.json"
    md_path = out_dir / f"{bench}.md"

    cfg = run_config or {}
    header = json.dumps({"_run_config": cfg})
    body = "\n".join(r.model_dump_json() for r in results)
    json_path.write_text(header + "\n" + (body + "\n" if body else ""), encoding="utf-8")

    cfg_md = ""
    if cfg:
        cfg_md = "\n".join(f"- {k}: `{v}`" for k, v in sorted(cfg.items())) + "\n\n"
    if results:
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results]
        md = (
            f"# {bench} eval — {results[0].model_id}\n\n"
            f"{cfg_md}"
            f"- examples: {len(results)}\n"
            f"- mean score: {statistics.mean(scores):.3f}\n"
            f"- median score: {statistics.median(scores):.3f}\n"
            f"- mean latency: {statistics.mean(latencies):.0f} ms\n"
        )
    else:
        md = f"# {bench} eval — no results\n\n{cfg_md}"
    md_path.write_text(md, encoding="utf-8")


@app.command()
def main(
    model: str = typer.Option(..., help="Model id label, used for the artifact dir."),
    bench: str = typer.Option(..., help="One of: factuality, refusal, extraction."),
    bench_file: Path = typer.Option(..., help="Path to the JSONL benchmark."),
    base_url: str = typer.Option(
        None,
        help="OpenAI-compatible URL. Defaults to $CIVIC_SLM_CANDIDATE_URL.",
    ),
    served_model: str = typer.Option(
        None,
        help="Model name the server expects. Defaults to $CIVIC_SLM_CANDIDATE_MODEL.",
    ),
    seed: int = typer.Option(0, help="Sampling seed; recorded in the run config."),
    temperature: float = typer.Option(
        0.0, help="Sampling temperature; recorded in the run config."
    ),
    max_tokens: int = typer.Option(512, help="Per-request max tokens; recorded in the run config."),
    similarity: str = typer.Option(
        "word_overlap",
        help=(
            "Factuality similarity scorer: 'word_overlap' (default; pre-v0.2 "
            "behavior, no extra deps) or 'bge' (BAAI/bge-large-en-v1.5 dual-"
            "encoder cosine; requires the `eval` extra). Switching changes "
            "the score scale — pre-v0.2 numbers are not comparable to BGE."
        ),
    ),
    bge_model: str = typer.Option(
        "BAAI/bge-large-en-v1.5",
        help="HF id of the dual-encoder when --similarity bge.",
    ),
    allow_contamination: bool = typer.Option(
        False,
        help=(
            "Skip the train/eval source-document contamination check. "
            "Use only with explicit reason."
        ),
    ),
) -> None:
    configure()
    base_url = base_url or runtimes.candidate_url()
    served_model = served_model or runtimes.candidate_model()
    examples = load_examples(bench_file)
    examples = [ex for ex in examples if ex.bench == bench]
    log.info("loaded_examples", bench=bench, count=len(examples), url=base_url)

    assert_no_contamination(
        examples,
        data_dir=settings().data_dir,
        allow_contamination=allow_contamination,
    )

    similarity_fn = _resolve_similarity(similarity, bge_model)

    client = ChatClient(
        base_url=base_url,
        model=served_model,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    results = run(examples=examples, client=client, model_id=model, similarity_fn=similarity_fn)

    out_dir = settings().artifacts_dir / "evals" / model
    run_config: dict[str, object] = {
        "model_id": model,
        "served_model": served_model,
        "base_url": base_url,
        "bench": bench,
        "bench_file": str(bench_file),
        "n_examples": len(examples),
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "similarity": similarity,
        "bge_model": bge_model if similarity == "bge" else None,
        "civic_slm_version": _resolve_version(),
    }
    write_report(results, out_dir, bench, run_config=run_config)
    log.info("eval_complete", out=str(out_dir), n=len(results))
    if results:
        typer.echo(json.dumps({"mean_score": statistics.mean(r.score for r in results)}))


def _resolve_version() -> str:
    from civic_slm import __version__

    return __version__


def _resolve_similarity(name: str, bge_model: str) -> Callable[[str, str], float] | None:
    """Map the `--similarity` CLI choice to a callable for `score_factuality`.

    `word_overlap` returns `None`, which makes the scorer fall back to its
    bundled word-overlap implementation. `bge` lazy-loads
    `sentence_transformers` from the `eval` extra and surfaces an actionable
    error if it isn't installed.
    """
    if name == "word_overlap":
        return None
    if name == "bge":
        try:
            from civic_slm.eval.embeddings import bge_similarity_fn
        except ImportError as exc:  # pragma: no cover — exercised via doctor
            raise typer.BadParameter(
                "--similarity bge requires the `eval` extra: `uv sync --extra eval`."
            ) from exc
        return bge_similarity_fn(bge_model)
    raise typer.BadParameter(f"--similarity must be one of: word_overlap, bge (got {name!r}).")


if __name__ == "__main__":
    app()
