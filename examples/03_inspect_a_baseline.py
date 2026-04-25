"""Read the committed base-Qwen baseline and print a human summary.

This script needs no server, no API key, no internet — only `uv sync`.
It's the lowest-friction way to confirm the eval harness is wired up
and to get a feel for what a baseline run looks like.

Quick start:

    uv run python examples/03_inspect_a_baseline.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

BASELINES = Path("artifacts/evals/base-qwen2.5-7b")


def main() -> None:
    if not BASELINES.exists():
        print(f"no baselines at {BASELINES} — committed baselines may have moved")
        return
    for json_path in sorted(BASELINES.glob("*.json")):
        bench = json_path.stem
        results: list[dict[str, object]] = []
        for line in json_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            # First line is _run_config (added in v0.1.0); skip it for stats.
            if "_run_config" in payload:
                continue
            results.append(payload)
        if not results:
            print(f"{bench:<22} no result rows")
            continue
        scores = [float(r["score"]) for r in results]  # type: ignore[arg-type]
        latencies = [float(r["latency_ms"]) for r in results]  # type: ignore[arg-type]
        print(
            f"{bench:<22} n={len(results):>3}  "
            f"mean={statistics.mean(scores):.3f}  "
            f"median={statistics.median(scores):.3f}  "
            f"latency~{statistics.mean(latencies):.0f}ms"
        )

    print("\nTo reproduce: see RELEASING.md → 'Eval baselines are still the published numbers'.")


if __name__ == "__main__":
    main()
