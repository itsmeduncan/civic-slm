"""Run the shipped factuality benchmark end-to-end against a locally-served model.

Same prerequisites as `01_ask_a_question.py`. Reads
`data/eval/civic_factuality.jsonl`, asks the model each question, scores
the answer, and prints a one-line summary. Does not write to `artifacts/`
unless you set `--save`.

Quick start:

    uv run python examples/02_run_factuality_eval.py

This is the same machinery `civic-slm eval run` uses, just with the
glue laid bare so you can read it.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from civic_slm.eval.runner import load_examples, run, write_report
from civic_slm.serve import runtimes
from civic_slm.serve.client import ChatClient


def main() -> None:
    save = "--save" in sys.argv
    bench_file = Path("data/eval/civic_factuality.jsonl")
    examples = [ex for ex in load_examples(bench_file) if ex.bench == "factuality"]
    print(f"→ loaded {len(examples)} factuality examples from {bench_file}")

    client = ChatClient(base_url=runtimes.candidate_url(), model=runtimes.candidate_model())
    print(f"→ asking {client.model} at {client.base_url}\n")
    results = run(examples=examples, client=client, model_id="example-run")

    scores = [r.score for r in results]
    print(f"mean score:   {statistics.mean(scores):.3f}")
    print(f"median score: {statistics.median(scores):.3f}")
    print(f"mean latency: {statistics.mean(r.latency_ms for r in results):.0f} ms")

    if save:
        out = Path("artifacts/evals/example-run")
        write_report(
            results,
            out,
            "factuality",
            run_config={
                "model_id": "example-run",
                "served_model": client.model,
                "base_url": client.base_url,
                "bench": "factuality",
                "n_examples": len(examples),
            },
        )
        print(f"\n→ wrote {out}/factuality.{{json,md}}")


if __name__ == "__main__":
    main()
