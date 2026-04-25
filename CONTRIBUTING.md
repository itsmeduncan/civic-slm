# Contributing

Thanks for picking this up. The project ships an open-source fine-tune of Qwen2.5-7B for U.S. local-government documents — cities, counties, townships, school districts across all 50 states. Everything from crawl to merge runs on a single Apple Silicon Mac. This doc covers how to get set up, what we expect from a contribution, and the tools we use. To add a new jurisdiction recipe specifically, see `docs/RECIPES.md`.

## Prerequisites

- macOS, Apple Silicon (M-series). Some stages (training, MLX serve, GGUF quantize) won't run on Linux/Intel.
- Python 3.11 (`uv` will install it for you if needed).
- `uv` for package management. Install: `brew install uv`.
- For the GGUF path: `brew install llama.cpp` (optional, only needed for `scripts/merge_quantize.py` and the 72B comparator).

## First-run setup

```bash
git clone https://github.com/itsmeduncan/civic-slm.git
cd civic-slm
uv sync                        # installs core + dev deps
uv run pytest                  # 65 tests, ~0.15s. Should be all green.
uv run civic-slm --help
```

If that all passes, you have a working dev environment. For training/synth/eval against real models, install the heavy extras:

```bash
uv sync --extra train --extra synth --extra eval --extra ingest
```

`train` and `eval` pull in MLX, transformers, sentence-transformers — large but Mac-native. `ingest` pulls in `browser-use` + `playwright`; you'll also need `playwright install chromium` once before crawling.

## Secrets

Project secrets live at `~/.config/civic-slm/.env` (not in the repo, not in any worktree). Format:

```
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
```

Use `civic_slm.config.require("HF_TOKEN")` to fetch — it raises an actionable error if missing.

## Running the test suite

```bash
uv run pytest                  # all tests
uv run pytest tests/test_schema.py -v
uv run pytest -k factuality    # name filter
```

Tests must stay fast (currently ~0.1s) and not require GPUs, network, or external secrets. If you need to test against a live model or API, mock it (see `tests/test_ingest.py` for the fake-fetcher pattern, or `tests/test_eval_runner.py` for the stub `ChatClient`).

### What's tested

- **schema** — round-trip JSON + reject-malformed for every Pydantic model.
- **config** — env loader, `require()` error path.
- **ingest** — end-to-end crawl with stub fetcher, manifest dedupe, idempotent re-run.
- **pdf** — section-aware chunker on synthetic text fixtures.
- **scorers** — factuality / refusal / extraction scorers with hand-built cases.
- **eval runner** — load JSONL → run → write report.
- **judge** — pairwise verdict parser (code fences, bad winner, garbage input).
- **synth** — JSONL parser, drops invalid lines, normalizes object outputs.
- **train** — command builders for CPT / SFT / DPO produce the right MLX-LM CLI.
- **backend** — LLM backend dispatch (`local` vs `anthropic`), env precedence, OpenAI-compatible payload shape via mocked transport.
- **caption** — VTT/SRT parsing, YouTube rolling-cue dedup, voice-tag and `>>` speaker preservation.
- **video_ingest** — `crawl_videos()` orchestrator with stubbed yt-dlp + ASR: idempotent re-runs, recipes without `discover_videos`, empty-transcript skip.
- **cli** — every umbrella subcommand (`crawl`, `crawl-videos`, `doctor`, `eval`, `train`, `version`) reachable via `--help`.
- **strict_local** — env-var parsing, `select_backend()` and `agent_llm()` raising on Anthropic-bound configs, `doctor --strict-local` exit codes. 21 cases covering every truthy/falsy spelling.

## Code style

Tooling: `ruff` for lint+format, `pyright` strict for typing, `pytest` for tests. All three must be clean before a PR lands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Conventions:

- `from __future__ import annotations` at the top of every Python file.
- Type hints on every public function. Pydantic v2 for any structured data crossing a stage boundary.
- 100-column lines, 4-space indent, double quotes (set by `ruff.toml`).
- Logging via `structlog`, not `print`. JSON in non-TTY contexts, pretty in TTY.
- Typer for CLIs. Every script runnable as `python -m civic_slm.<module>`.
- No LangChain. Anthropic SDK direct.
- Lazy-import optional heavy deps (`pypdf`, `browser_use`, `mlx_lm`, `wandb`, `anthropic`) at use sites so the core package stays installable.
- Docstrings explain _why_, not _what_. Skip them for trivial functions.

## Working on a stage

The core discipline: **eval first, training last**. If you change anything in `src/civic_slm/eval/` or `data/eval/`, re-run the baselines and confirm the numbers in `artifacts/evals/base-qwen2.5-7b/` are reproducible. A scorer change that silently shifts the floor will mask real model regressions.

```bash
# terminal 1
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# terminal 2 — re-run baseline
uv run civic-slm eval run \
    --model base-qwen2.5-7b \
    --bench factuality \
    --bench-file data/eval/civic_factuality.jsonl \
    --base-url http://127.0.0.1:8080 \
    --served-model mlx-community/Qwen2.5-7B-Instruct-4bit
```

For long training runs, always do the dry-run first:

```bash
uv run civic-slm train cpt --max-iters 100 --dry-run    # prints the command
uv run civic-slm train cpt --max-iters 100               # actually runs 100 steps
```

100 steps confirms loss drops and memory stays in budget. Only then commit to the full run.

## Pull request workflow

1. Branch off `main` with a descriptive name: `feature/synth-grounding-check`, `fix/refusal-regex-edge-case`.
2. Make your change. Keep diffs focused — one stage per PR when possible.
3. Run the full check before pushing:
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
   ```
4. Update `CHANGELOG.md` under `[Unreleased]` with a user-forward entry. Lead with what someone can now _do_, not what was changed internally. Internal-only changes go under `### For contributors`.
5. If it ships a new artifact (eval report, model version, dataset), bump `VERSION` and add a release entry.
6. **Sign off your commits.** This project uses the [Developer Certificate of Origin](https://developercertificate.org/) — every commit must carry a `Signed-off-by: Your Name <you@example.com>` trailer. The easiest way is `git commit -s`, which adds it automatically. Signing off is a lightweight statement that you have the right to contribute the code under the project's license; it is not a CLA. If you forget, amend with `git commit --amend -s --no-edit` (or for older commits in your branch, `git rebase --signoff main`).
7. Open the PR. The PR body should describe the user-visible change and link to the failing baseline (if applicable). For releases, see `RELEASING.md`.

## Filing an issue

Useful issues include:

- A specific repro for a bug (input → expected → actual).
- Eval results that look wrong, with `artifacts/evals/.../{bench}.json` attached.
- Crawl recipes that fail on a specific city, with a transcript or screenshot.

## Architecture decisions

If you're proposing something that changes the contract — schema fields, eval scoring, training stages, the `Recipe` Protocol — read `ARCHITECTURE.md` first and call out which decision you're revisiting in the PR description. Architecture is rarely the bottleneck; usually the better answer is "make the existing seam work for your case."

## License

MIT — see `LICENSE`.
