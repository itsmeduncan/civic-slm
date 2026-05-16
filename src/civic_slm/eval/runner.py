"""Eval runner: loads a benchmark JSONL, asks the model, scores, writes a report.

CLI: `python -m civic_slm.eval.runner --model qwen-civic-base --bench factuality \\
        --base-url http://localhost:8080 --bench-file data/eval/civic_factuality.jsonl`
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import typer
from pydantic import TypeAdapter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
from civic_slm.serve import models, runtimes
from civic_slm.serve.client import ChatClient, ChatResponse


class ContaminationError(RuntimeError):
    """Raised when an eval example's source document is also in the train manifest."""


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

log = get_logger(__name__)
app = typer.Typer(help="Run an evaluation benchmark against a served model.")

EVAL_ADAPTER: TypeAdapter[EvalExample] = TypeAdapter(EvalExample)


_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _chat_with_retry(
    client: ChatClient, system: str, user: str, *, max_attempts: int = 3
) -> ChatResponse:
    """Call `client.chat` with exponential backoff on transient HTTP errors.

    Without this, a single 429 or 503 from LM Studio mid-bench charges the
    failing example a score=0 and the maintainer has to re-run 1/200 by hand.
    We retry only on classes that can realistically heal between attempts:
    HTTP 408/425/429/5xx, `ReadTimeout`, and `RemoteProtocolError` (mid-
    stream connection drops). `ConnectError` (server not running) is NOT
    retried — backing off three times against a refused socket just delays
    the inevitable failure and hides the real problem from the operator.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    delay = 1.0
    last: BaseException = RuntimeError("unreachable: retry loop exited without an attempt")
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat(system, user)
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code not in _TRANSIENT_STATUS_CODES:
                raise
        except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            last = exc
        if attempt < max_attempts:
            log.info(
                "chat_retry",
                attempt=attempt,
                error=type(last).__name__,
                sleep_s=delay,
            )
            time.sleep(delay)
            delay *= 2
    raise last


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


def eval_progress(description: str) -> Progress:
    """Rich progress bar with elapsed/ETA, suited to per-example LLM calls."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def run(
    *,
    examples: list[EvalExample],
    client: ChatClient,
    model_id: str,
    similarity_fn: Callable[[str, str], float] | None = None,
    progress_label: str = "eval",
) -> list[EvalResult]:
    """Run an evaluation. `similarity_fn` is forwarded to `score_factuality`.

    `None` keeps the legacy word-overlap behavior so pre-v0.2 baselines remain
    reproducible. Pass `bge_similarity_fn()` (from `civic_slm.eval.embeddings`)
    to use the BGE dual-encoder cosine.
    """
    results: list[EvalResult] = []
    progress = eval_progress(progress_label)
    with progress:
        task = progress.add_task(progress_label, total=len(examples))
        for ex in examples:
            # Per-example exception isolation: a single ReadTimeout (or any
            # transient HTTP error) used to kill the whole bench, losing every
            # completed example. Now we record a failure marker (score=0,
            # judge_notes=error class) and continue — the bench finishes, the
            # report shows which examples broke, and the maintainer can re-run
            # those specific ones rather than the full 200.
            try:
                if isinstance(ex, FactualityExample):
                    user = f"Context:\n{ex.context}\n\nQuestion: {ex.question}"
                    resp = _chat_with_retry(client, _FACTUALITY_SYSTEM, user)
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
                    resp = _chat_with_retry(client, _FACTUALITY_SYSTEM, user)
                    results.append(
                        score_refusal(ex, resp.text, model_id=model_id, latency_ms=resp.latency_ms)
                    )
                elif isinstance(ex, ExtractionExample):
                    user = (
                        f"Schema: {ex.schema_name}\n\n"
                        f"Document:\n{ex.document_text}\n\nReturn the JSON now."
                    )
                    resp = _chat_with_retry(client, _EXTRACTION_SYSTEM, user)
                    results.append(
                        score_extraction(
                            ex, resp.text, model_id=model_id, latency_ms=resp.latency_ms
                        )
                    )
                else:
                    # SideBySideExample requires a pairwise judge — separate runner.
                    log.info("skipping_side_by_side", id=ex.id)
                    progress.advance(task)
                    continue
            except Exception as exc:
                bench_kind = (
                    "factuality"
                    if isinstance(ex, FactualityExample)
                    else "refusal"
                    if isinstance(ex, RefusalExample)
                    else "extraction"
                )
                log.warning(
                    "example_failed",
                    bench=bench_kind,
                    example_id=ex.id,
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )
                results.append(
                    EvalResult(
                        model_id=model_id,
                        bench=bench_kind,
                        example_id=ex.id,
                        prediction="",
                        score=0.0,
                        judge_notes=f"FAILED: {type(exc).__name__}: {str(exc)[:200]}",
                        latency_ms=0.0,
                    )
                )
            progress.advance(task)
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
    model: str = typer.Option(
        ...,
        help=(
            "Project-side model label (e.g. base-qwen3.6-27b). Resolves through "
            "civic_slm.serve.models.MODELS to BOTH the artifact directory and the "
            "served-model name sent to LM Studio — they cannot disagree. Unregistered "
            "labels are passed through verbatim."
        ),
    ),
    bench: str = typer.Option(..., help="One of: factuality, refusal, extraction."),
    bench_file: Path = typer.Option(..., help="Path to the JSONL benchmark."),
    base_url: str = typer.Option(
        None,
        help="OpenAI-compatible URL. Defaults to $CIVIC_SLM_LM_STUDIO_URL.",
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
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help=(
            "Allow the served model to emit hidden chain-of-thought tokens "
            "(reasoning content). Defaults to OFF on factuality / refusal / "
            "extraction because the scorer never reads reasoning tokens and "
            "they tax latency 5-10x on Qwen 3.6 reasoning models. Pass "
            "--thinking to re-enable per run (e.g. to confirm a score doesn't "
            "depend on CoT). Implemented as `chat_template_kwargs="
            "{'enable_thinking': false}` on the request, which LM Studio and "
            "llama-server both pass through to the Jinja template."
        ),
    ),
) -> None:
    configure()
    runtimes.assert_no_deprecated_env()
    resolved = models.resolve(model)
    base_url = base_url or runtimes.lm_studio_url()
    examples = load_examples(bench_file)
    total_loaded = len(examples)
    examples = [ex for ex in examples if ex.bench == bench]
    if total_loaded and not examples:
        # Easy mistake: `--bench refusal --bench-file civic_factuality.jsonl`
        # silently produces an empty run. Warn so the operator notices.
        log.warning(
            "no_examples_after_bench_filter",
            bench=bench,
            bench_file=str(bench_file),
            total_loaded=total_loaded,
        )
    log.info(
        "loaded_examples",
        bench=bench,
        count=len(examples),
        model_label=resolved.label,
        served_name=resolved.served_name,
        url=base_url,
        thinking=thinking,
    )

    assert_no_contamination(
        examples,
        data_dir=settings().data_dir,
        allow_contamination=allow_contamination,
    )

    similarity_fn = _resolve_similarity(similarity, bge_model)

    chat_template_kwargs: dict[str, object] | None = (
        None if thinking else {"enable_thinking": False}
    )
    client = ChatClient(
        base_url=base_url,
        model=resolved.served_name,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    results = run(
        examples=examples,
        client=client,
        model_id=resolved.label,
        similarity_fn=similarity_fn,
        progress_label=f"eval {bench}",
    )

    out_dir = settings().artifacts_dir / "evals" / resolved.label
    run_config: dict[str, object] = {
        "model_label": resolved.label,
        "served_name": resolved.served_name,
        "base_url": base_url,
        "bench": bench,
        "bench_file": str(bench_file),
        "n_examples": len(examples),
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "similarity": similarity,
        "bge_model": bge_model if similarity == "bge" else None,
        "thinking": thinking,
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
