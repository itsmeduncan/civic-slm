# Civic SLM

> **v0.1.0 — infrastructure preview.** No fine-tuned model has shipped yet; what's in this repo is the pipeline that will produce one. Baselines for `Qwen/Qwen2.5-7B-Instruct` are committed under `artifacts/evals/base-qwen2.5-7b/`. The "all 50 states" framing is the design target — the only registered recipe today is `san-clemente`, and it requires a per-source license audit (`docs/SOURCES.md`) before its first real crawl. See [`MODEL_CARD.md`](MODEL_CARD.md), [`DATA_CARD.md`](DATA_CARD.md), [`ACCEPTABLE_USE_POLICY.md`](ACCEPTABLE_USE_POLICY.md), and [`ROADMAP.md`](ROADMAP.md) for the honest state of things.

A domain-specialized fine-tune of **Qwen2.5-7B-Instruct** for **U.S. local-government documents** — city, county, and township agendas, staff reports, comprehensive plans, minutes, ordinances, and municipal codes. Designed to power civic transparency tools across all 50 states.

Trained on a single Apple Silicon Mac via [MLX-LM](https://github.com/ml-explore/mlx). **Served on whatever runtime you like** — MLX, [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), [llama.cpp](https://github.com/ggerganov/llama.cpp), or any OpenAI-compatible endpoint. Released as both **MLX-q4** and **GGUF Q5_K_M**. Documents crawled with [browser-use](https://github.com/browser-use/browser-use) — one recipe per jurisdiction, recipes are tiny.

- **Add a jurisdiction:** [docs/RECIPES.md](docs/RECIPES.md) (San Clemente, CA ships as the demo).
- **Pick a runtime:** [docs/RUNTIMES.md](docs/RUNTIMES.md) (1-model minimum setup, copy-paste for each runtime).

## Pipeline

```
crawl ──► chunk ──► synthesize ──► CPT ──► SFT ──► DPO ──► merge + quantize ──► eval
   │         │            │          │       │       │            │              │
   │         │            │          └───────┴───────┴────────────┴──► W&B run logs
   │         │            └──► Anthropic SDK or local LLM (env-switchable)
   │         └──► Pydantic-validated DocumentChunks
   └──► browser-use recipe per jurisdiction (any U.S. city/county/township)
```

## Quickstart

```bash
uv sync --all-extras
uv run pytest                                    # 65 tests across schema, ingest, scorers, synth, train, llm-backend, video/caption
uv run civic-slm --help
```

For an end-to-end walkthrough — crawl → synth → train → eval → release — see [docs/USAGE.md](docs/USAGE.md). To add a new jurisdiction, see [docs/RECIPES.md](docs/RECIPES.md).

Secrets live at `~/.config/civic-slm/.env`:

```
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
```

The Anthropic key is optional — set `CIVIC_SLM_LLM_BACKEND=local` to run synth, the side-by-side judge, and the crawler against a local OpenAI-compatible endpoint (e.g. Qwen2.5-72B served via `llama-server`).

**Want to run zero paid tokens, with proof?** Set `CIVIC_SLM_STRICT_LOCAL=1` alongside `=local` and run `civic-slm doctor --strict-local`. The tripwire makes synth, judge, and crawler **refuse** to call Anthropic at runtime, and the doctor exits non-zero if any code path could reach a paid endpoint. See [docs/RUNTIMES.md#strict-local-mode](docs/RUNTIMES.md#strict-local-mode-zero-api-spend-with-proof).

## CLI

The `civic-slm` umbrella exposes every stage:

```
civic-slm doctor                                  # sanity-check secrets + runtime
civic-slm crawl --jurisdiction san-clemente --max 20
civic-slm crawl-videos --jurisdiction san-clemente --max 20      # YouTube meeting recordings → transcript
civic-slm eval run --model <id> --bench factuality --bench-file data/eval/civic_factuality.jsonl
civic-slm eval side-by-side --candidate-model <id>
civic-slm train cpt | sft | dpo --config configs/<stage>.yaml [--dry-run] [--max-iters 100]
```

Synthetic SFT review (terminal accept/reject loop): `python scripts/review_sft.py`.
Merge + quantize a final adapter: `python scripts/merge_quantize.py`.

## Eval-first

The training contract is **no training without a baseline**. The four benchmarks in `data/eval/` run against base Qwen2.5-7B before any fine-tuning starts; those numbers are what every subsequent stage has to beat.

| Bench                   | What it measures                                     | Score                                                  |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| `civic_factuality`      | Q&A grounded in held-out docs                        | citation exact-match + word-overlap (BGE swap planned) |
| `refusal`               | refuses when context lacks the answer                | refusal rate (regex + fallback judge)                  |
| `structured_extraction` | staff report → JSON                                  | field-level F1                                         |
| `side_by_side`          | open-ended U.S. municipal prompts vs base 7B and 72B | Claude or local-LLM judge w/ A/B position swap         |

Current example counts are the v0 bootstrap (10 / 10 / 5 / 10). Target sizes for v1 per the training contract: **200 / 100 / 50 / 100**.

### Run a baseline

Pick any OpenAI-compatible runtime — [docs/RUNTIMES.md](docs/RUNTIMES.md) covers MLX, Ollama, LM Studio, and llama.cpp. The shortest path:

```bash
# terminal 1 — start any runtime serving Qwen2.5-7B-Instruct-4bit
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# terminal 2 — sanity-check, then run the bench
uv run civic-slm doctor
uv run civic-slm eval run \
    --model base-qwen2.5-7b \
    --bench factuality \
    --bench-file data/eval/civic_factuality.jsonl
```

If you're using something other than MLX on port 8080, point civic-slm at it:

```bash
export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:11434     # Ollama
export CIVIC_SLM_CANDIDATE_MODEL=qwen2.5:7b-instruct-q4_K_M
```

Reports land at `artifacts/evals/<model_id>/<bench>.{json,md}`.

## Baseline numbers (Qwen2.5-7B-Instruct 4-bit, MLX)

| Bench        | n   | Mean                       | Median | Latency |
| ------------ | --- | -------------------------- | ------ | ------- |
| factuality   | 10  | 0.501                      | 0.566  | 637 ms  |
| refusal      | 10  | 0.800                      | 1.000  | 460 ms  |
| extraction   | 5   | 0.277                      | 0.000  | 925 ms  |
| side_by_side | —   | — (pending 72B comparator) | —      | —       |

These are the bars the fine-tune has to clear. Refusal is already strong on the base model — protect it. Extraction is the biggest training opportunity.

## Status

Scaffold, schemas, ingestion (browser-use + San Clemente demo recipe + a recipe template for any U.S. jurisdiction), 4-bench eval harness, synth pipeline (Anthropic _or_ fully-local LLM backend), MLX training scripts (CPT/SFT/DPO), merge+quantize to MLX-q4 + GGUF Q5_K_M, runtime-agnostic serving (MLX / Ollama / LM Studio / llama.cpp), and committed baselines for factuality / refusal / extraction. Next: synth corpus + first training pass.

See `CLAUDE.md` for the full project contract, `ARCHITECTURE.md` for design decisions, `docs/RUNTIMES.md` for picking a runtime, and `docs/RECIPES.md` for adding new jurisdictions.
