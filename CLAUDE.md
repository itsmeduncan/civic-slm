# Civic SLM: Qwen Fine-Tune for Local Government Intelligence

You are helping build an open-source small language model specialized for **U.S. local-government** document understanding (cities, counties, townships, school districts — all 50 states). The model is a LoRA fine-tune of Qwen2.5-7B-Instruct, trained on multi-jurisdiction civic corpora (comprehensive/general/master plans, staff reports, meeting minutes, ordinances, municipal codes), designed to power civic transparency tools.

San Clemente, CA is the demo recipe; the architecture is intentionally extensible to any U.S. jurisdiction (see `docs/RECIPES.md`).

## Project goals

1. Produce a merged, quantized SLM that decisively outperforms base Qwen2.5-7B on civic tasks and approaches Qwen2.5-72B performance on domain-specific benchmarks.
2. Generate training data, eval harnesses, and training configs that are reproducible and auditable.
3. Ship artifacts suitable for open-source release: weights (HF Hub), dataset (HF Datasets), eval results, model card.

## Environment

- **Host: macOS, Apple Silicon, single machine.** All ingestion, synthesis, training, eval, and serving run locally on this Mac. Unified memory budget governs model size choices.
- Python 3.11, `uv` for package management.
- **Frameworks**: **MLX-LM** for training (LoRA, DPO) and in-process inference; **llama.cpp** (`llama-server`) for OpenAI-compatible HTTP serving and the 72B GGUF comparator.
- **Crawling**: real browsers driven via [`browser-use`](https://github.com/browser-use/browser-use) / [`browser-harness`](https://github.com/browser-use/browser-harness). No platform-specific scrapers (no hand-written Granicus/Legistar/CivicPlus/Municode logic) — recipes are LLM-driven instructions per jurisdiction. One recipe template (`recipes/_template.py`) covers any U.S. city, county, or township regardless of vendor.
- Storage: `~/Projects/src/github.com/itsmeduncan/civic-slm/` as project root; HF cache at default `~/.cache/huggingface/`.
- Secrets in `~/.config/civic-slm/.env` (`HF_TOKEN`, `ANTHROPIC_API_KEY`, `WANDB_API_KEY`).
- Runtime config via env vars (see `docs/RUNTIMES.md` for the full table): `CIVIC_SLM_LLM_BACKEND` (`local|anthropic`), `CIVIC_SLM_CANDIDATE_URL`/`_MODEL`, `CIVIC_SLM_TEACHER_URL`/`_MODEL`, `CIVIC_SLM_STRICT_LOCAL` (tripwire that refuses Anthropic), `CIVIC_SLM_TIMEOUT_S`, `CIVIC_SLM_WHISPER_MODEL`.

## Repository layout to create and maintain

```
civic-slm/
├── pyproject.toml
├── README.md
├── CHANGELOG.md           # Keep-a-Changelog
├── ARCHITECTURE.md        # design decisions
├── CONTRIBUTING.md        # dev workflow
├── LICENSE                # MIT
├── VERSION                # source of truth for releases
├── docs/
│   ├── USAGE.md           # end-to-end walkthrough
│   ├── RECIPES.md         # add a new U.S. jurisdiction
│   └── RUNTIMES.md        # serve via MLX / Ollama / LM Studio / llama.cpp
├── configs/
│   ├── cpt.yaml           # continued pretraining (MLX)
│   ├── sft.yaml           # instruction tuning (MLX)
│   └── dpo.yaml           # preference optimization (MLX)
├── data/
│   ├── raw/               # crawled documents (gitignored, manifest tracked)
│   ├── processed/         # cleaned, chunked
│   ├── sft/               # instruction pairs (jsonl)
│   ├── dpo/               # preference pairs (jsonl)
│   └── eval/              # held-out benchmarks (jsonl)
├── src/civic_slm/
│   ├── cli.py             # umbrella Typer (crawl, eval, train, doctor, version)
│   ├── doctor.py          # `civic-slm doctor` — env + runtime sanity check
│   ├── ingest/            # PDF + video crawlers, recipes (incl. _template.py, _youtube.py)
│   │   ├── recipes/       # one file per jurisdiction; _template + _youtube + _browser helpers
│   │   └── video/         # caption, youtube (yt-dlp), transcript, asr (mlx-whisper)
│   ├── synth/             # synthetic data generation (backend-agnostic)
│   ├── train/             # MLX-LM training wrappers + dataset.py iters helper
│   ├── eval/              # benchmark runners + judge
│   ├── llm/               # backend abstraction (anthropic | local OpenAI-compatible)
│   └── serve/             # ChatClient + runtime presets / helpers
├── scripts/               # one-off CLI entry points (merge_quantize, review_sft)
├── notebooks/             # exploration only, not source of truth
└── tests/                 # pytest, fast unit tests on data pipelines
```

## Coding standards

- Type hints on all public functions. `from __future__ import annotations`.
- Pydantic v2 for any structured data contracts (agenda items, staff reports, training examples).
- Typer for CLI entry points. Every script runnable as `python -m civic_slm.<module>`.
- Logging via `structlog`, not print. JSON logs in production paths.
- No LangChain. Use the Anthropic SDK directly for synthetic data generation.
- No notebooks in the commit path for anything that matters. If it ships, it's in `src/`.
- `ruff` for lint + format, `pyright` strict for typing, `pytest` with coverage.

## Training pipeline contract

Stages execute in order, each producing a versioned artifact:

1. **CPT**: 1-2 epochs on raw civic corpus, LR 1e-5, cosine, LoRA r=64, all linear layers. Output: `artifacts/qwen-civic-cpt/`.
2. **SFT**: 3 epochs on instruction pairs, LR 2e-4, warmup 3%, packing enabled, LoRA r=32 α=64. Output: `artifacts/qwen-civic-sft/`.
3. **DPO**: 1 epoch on preference pairs, LR 5e-7, β=0.1. Output: `artifacts/qwen-civic-dpo/`. (If MLX-LM DPO support is too rough, ship v0 as CPT+SFT and revisit DPO in v1.)
4. **Merge + quantize**: fuse final adapter into the base, export **MLX 4-bit** (primary Mac artifact) and **GGUF Q5_K_M** (llama.cpp / Ollama users). Output: `artifacts/qwen-civic-v{N}-mlx-q4/`, `artifacts/qwen-civic-v{N}-gguf-q5km/`.

Every stage logs to W&B under project `civic-slm`, with run names `{stage}-{git_sha}-{timestamp}`.

## Evaluation contract (build BEFORE training)

Four benchmarks, all in `data/eval/`, all runnable via `python -m civic_slm.eval.run --model <path> --bench <name>`:

1. `civic_factuality.jsonl` — Q&A pairs, answer provable from held-out doc. Score: citation exact-match + answer semantic similarity (BGE reranker as judge). Start with 10 hand-written, grow toward 200.
2. `refusal.jsonl` — adversarial questions where context does NOT contain the answer. Score: refusal rate (must decline, not confabulate). Start with 10, grow toward 100.
3. `structured_extraction.jsonl` — staff reports with ground-truth JSON. Score: field-level F1. Start with 5, grow toward 50.
4. `side_by_side.jsonl` — prompts compared against base Qwen2.5-7B and Qwen2.5-72B (both run locally — 7B as MLX 4-bit, 72B as GGUF Q4 via llama.cpp) via pairwise LLM-judge (Claude Sonnet 4.6 as judge). Start with 10, grow toward 100.

Results emit to `artifacts/evals/{model_version}/{bench}.json` and a markdown report.

## Data generation approach

Synthetic instruction pairs via Claude Opus 4.7 using real civic documents as seed context. Pipeline:

1. Crawl real docs (start with San Clemente, CA; expand to a geographically diverse mix of U.S. jurisdictions across regions and platforms) using `browser-use`/`browser-harness`. Recipes per jurisdiction live in `src/civic_slm/ingest/recipes/` — one file per jurisdiction, copied from `_template.py`.
2. For each doc chunk, prompt Claude to generate (task, input, output) triples across the task taxonomy (summarization, extraction, grounded Q&A, refusal, diff analysis).
3. Human-review the first 500 examples, then bootstrap: use v0 model to generate candidates, human curates.
4. All examples validated against Pydantic schema before landing in `data/sft/`.

## Working agreements

- Always read SKILL.md files before creating files of that type. Before running any code, check for relevant skills.
- When implementing a stage, write the eval first, then the training code. TDD applies.
- Every non-trivial function gets a docstring explaining _why_, not _what_.
- Before running long training jobs, do a dry-run at 100 steps with `max_steps=100, logging_steps=10` to verify loss decreases and memory stays in budget.
- If you hit an architectural decision, pause and present 2-3 options with tradeoffs before committing. Do not quietly pick.
- Commit after every working stage. Conventional commits (`feat:`, `fix:`, `chore:`). Don't commit model weights; use HF Hub or local `artifacts/` (gitignored).
- If unified memory is tight, reduce batch size and increase gradient accumulation before reducing model quality (rank, precision). For inference with 72B comparator, run candidate and comparator sequentially per example, not concurrently.

## Project status

v0.1.0 shipped as an "infrastructure preview." v0.2.x has now landed all four
code-only tracks toward the v1 fine-tune (PR #6 BGE scorer, PR #7 training
supervisor + resume + smoke, PR #8 72B comparator wiring, PR #9 eval scale-up

- multi-jurisdiction seeding). Eval harness, synth pipeline, MLX training
  scripts, and merge+quantize are all in place. Bench sizes grew from 10/14/5/10
  to 25/29/15/25 — original v0 baselines are no longer comparable and
  `MODEL_CARD.md` shows _re-baselining_ pending a maintainer eval run against
  the served base model.

| Bench        | n (current) | Status           | Notes                                                                 |
| ------------ | ----------- | ---------------- | --------------------------------------------------------------------- |
| factuality   | 25          | re-baselining    | scorer accepts `--similarity {word_overlap,bge}` (BGE opt-in v0.2)    |
| refusal      | 29          | re-baselining    | 17 should-refuse + 12 should-answer; multi-jurisdiction               |
| extraction   | 15          | re-baselining    | multi-jurisdiction `staff_report` schema examples                     |
| side_by_side | 25          | comparator wired | needs Qwen2.5-72B GGUF on disk to actually run (see docs/RUNTIMES.md) |

The fine-tune has to clear these once they're re-baselined. **Do not start
training until the eval harness still produces these baselines** — regressions
in the scorer or the runner could mask real model gains.

### Next stages, in order

The four code-only prerequisites above are now in place. The remaining work is
maintainer-blocking — fixed costs in dev time, API spend, and HF/HW resources.

1. **Re-baseline base Qwen** on the new bench. With `mlx_lm.server` running:
   `civic-slm eval run --model base-qwen2.5-7b --bench {factuality,refusal,extraction}`.
   ~30 min wall-clock. Commit results to `artifacts/evals/base-qwen2.5-7b/`.
2. Fill `docs/SOURCES.md#san-clemente` with a verbatim ToS quote and flip the
   audit to GO. ~30 min.
3. Crawl real corpus: `civic-slm crawl --jurisdiction san-clemente --max 50`
   (PDFs) + `civic-slm crawl-videos --jurisdiction san-clemente --max 20`
   (transcripts via caption-first → Whisper fallback). ~½ day.
4. Generate synthetic SFT pairs by calling `civic_slm.synth.generate.generate_corpus()`
   from a Python entry point (no CLI wrapper yet — wire one in v0.3 or invoke
   directly: `python -c "import asyncio; from civic_slm.synth.generate import generate_corpus; asyncio.run(generate_corpus(...))"`).
   Human-review the first 500 with `python scripts/review_sft.py`. ~$5–15 in API credits.
5. CPT smoke (`civic-slm train cpt --smoke-test`) → full CPT → SFT → DPO. The
   trainer wrappers now propagate SIGTERM/SIGINT and refuse to overwrite an
   existing adapter without `--resume` (PR #7).
6. `python scripts/merge_quantize.py --version v1` → push to HF Hub → tag
   v0.2.0 per `RELEASING.md`.
7. Re-run all four benches after each stage; gate to next stage = beats prior
   version on ≥ 3/4 benches.

## Out of scope

- RAG serving infrastructure (separate project; this is the model only).
- Frontend / UI.
- Production deployment beyond local MLX / llama.cpp.
- Multi-machine training (single-Mac constraint is deliberate — forces discipline).

Ask clarifying questions if a decision will be hard to reverse. Otherwise, proceed and show your work.
