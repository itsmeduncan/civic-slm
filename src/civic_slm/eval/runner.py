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

if TYPE_CHECKING:
    from collections.abc import Iterator

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
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for ex in examples:
        if isinstance(ex, FactualityExample):
            user = f"Context:\n{ex.context}\n\nQuestion: {ex.question}"
            resp = client.chat(_FACTUALITY_SYSTEM, user)
            results.append(
                score_factuality(ex, resp.text, model_id=model_id, latency_ms=resp.latency_ms)
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


def write_report(results: list[EvalResult], out_dir: Path, bench: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{bench}.json"
    md_path = out_dir / f"{bench}.md"

    json_path.write_text(
        "\n".join(r.model_dump_json() for r in results) + "\n",
        encoding="utf-8",
    )

    if results:
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results]
        md = (
            f"# {bench} eval — {results[0].model_id}\n\n"
            f"- examples: {len(results)}\n"
            f"- mean score: {statistics.mean(scores):.3f}\n"
            f"- median score: {statistics.median(scores):.3f}\n"
            f"- mean latency: {statistics.mean(latencies):.0f} ms\n"
        )
    else:
        md = f"# {bench} eval — no results\n"
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
) -> None:
    configure()
    base_url = base_url or runtimes.candidate_url()
    served_model = served_model or runtimes.candidate_model()
    examples = load_examples(bench_file)
    examples = [ex for ex in examples if ex.bench == bench]
    log.info("loaded_examples", bench=bench, count=len(examples), url=base_url)

    client = ChatClient(base_url=base_url, model=served_model)
    results = run(examples=examples, client=client, model_id=model)

    out_dir = settings().artifacts_dir / "evals" / model
    write_report(results, out_dir, bench)
    log.info("eval_complete", out=str(out_dir), n=len(results))
    if results:
        typer.echo(json.dumps({"mean_score": statistics.mean(r.score for r in results)}))


if __name__ == "__main__":
    app()
