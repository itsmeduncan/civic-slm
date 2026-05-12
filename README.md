# Civic SLM

v0.2.0 — infrastructure preview. All code-only tracks for the v1 fine-tune are now landed (BGE scorer, training supervisor, side-by-side comparator, and multi-jurisdiction eval scale-up). The fine-tune base is **Qwen 3.6 27B** (served locally via LM Studio as `qwen3.6-27b-ud-mlx`); a re-baseline against that model is the next gate before training (see issue [#17](https://github.com/itsmeduncan/civic-slm/issues/17)). Historical Qwen 2.5 7B baselines remain at `artifacts/evals/base-qwen2.5-7b/` for reference. The "all 50 states" framing is the design target — the only registered recipe today is `san-clemente`, and it requires a per-source license audit (`docs/SOURCES.md`) before its first real crawl. See [`MODEL_CARD.md`](MODEL_CARD.md), [`DATA_CARD.md`](DATA_CARD.md), and [`ACCEPTABLE_USE_POLICY.md`](ACCEPTABLE_USE_POLICY.md) for the honest state of things.

## Why this exists

A 200-page general plan, a 50-page staff report, a municipal code that
sprawls across hundreds of sections — these are the documents that
govern what gets built, what gets funded, and what gets enforced in U.S.
cities and counties. They are public, but they are not _accessible_.
General-purpose chat models read them poorly: they hallucinate ordinance
numbers, invent fiscal-impact figures, and refuse to cite. A small,
auditable, domain-specialized model — one a journalist or a Code-for-
America brigade can run on a laptop — closes that gap without sending
constituent questions to a third party. That is the model this project
ships.

`civic-slm` is a domain-specialized fine-tune of **Qwen 3.6 27B Instruct**
for **U.S. local-government documents** — city, county, and township
agendas, staff reports, comprehensive plans, minutes, ordinances, and
municipal codes.

## Why fine-tune instead of base Qwen + RAG?

The honest answer is "do both, but they solve different problems." RAG
is what tells the model _which document_ to read; the fine-tune is what
teaches it _how civic documents are structured_ — that staff reports
have a fiscal-impact paragraph, that "CUP 24-031" is a file number not
a vote count, that "exempt under CEQA §15061(b)(3)" is a legal status
rather than something to summarize. Three concrete differences a fine-
tune buys you that RAG doesn't:

1. **Citation discipline.** Base Qwen will paraphrase and drop
   citations. The SFT corpus is built around grounded Q&A pairs that
   require citing item numbers and section names verbatim.
2. **Refusal calibration.** Base Qwen will confabulate when the answer
   is not in context. The refusal benchmark + DPO stage exist to push
   the model to decline when grounded.
3. **Structured extraction.** Base Qwen nests JSON under
   `staff_report` keys and improvises field names (extraction baseline
   is 0.277 — see below). The SFT corpus targets a flat, predictable
   schema.

This repo ships the model only — RAG is a separate concern. Pair the
released weights with whatever retrieval stack you already have.

Trained on a single Apple Silicon Mac via [MLX-LM](https://github.com/ml-explore/mlx). **Served on whatever runtime you like** — MLX, [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), [llama.cpp](https://github.com/ggerganov/llama.cpp), or any OpenAI-compatible endpoint. Released as both **MLX-q4** and **GGUF Q5_K_M**. Documents crawled with [browser-use](https://github.com/browser-use/browser-use) — one recipe per jurisdiction, recipes are tiny.

## What "done" looks like

`civic-slm v1` ships when the merged + quantized model beats base
base Qwen 3.6 27B on **at least 3 of 4** benchmarks at v1 sample sizes
(200 / 100 / 50 / 100 — see `MODEL_CARD.md`). The release also requires
a positively-confirmed source-license audit per recipe (`docs/SOURCES.md`)
and at least one second-city held-out eval (e.g., Austin TX) to back the
all-50-states framing. v1.0 commits to backward-compatibility per
`RELEASING.md`. Anything else is a v0.x preview.

- **First 5 minutes:** [`examples/`](examples/) — copy-paste demos. `03_inspect_a_baseline.py` runs without a server.
- **Add a jurisdiction:** [docs/RECIPES.md](docs/RECIPES.md) (San Clemente, CA ships as the demo).
- **Pick a runtime:** [docs/RUNTIMES.md](docs/RUNTIMES.md) (1-model minimum setup, copy-paste for each runtime).
- **Plain-language definitions:** [docs/GLOSSARY.md](docs/GLOSSARY.md) for the ML and civic terms used throughout.

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
uv run pytest                                    # ~107 tests across schema, ingest, scorers, synth, train, llm-backend, video/caption, eval-embeddings, train-supervisor, side-by-side
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
civic-slm crawl san-clemente --max 20
civic-slm crawl-videos san-clemente --max 20      # YouTube meeting recordings → transcript
civic-slm process san-clemente                                    # raw PDFs → data/processed/{jurisdiction}.jsonl
civic-slm synth san-clemente                                      # processed chunks → data/sft/{jurisdiction}.jsonl
civic-slm eval run --model <id> --bench factuality --bench-file data/eval/civic_factuality.jsonl
civic-slm eval side-by-side --candidate-model <id>
civic-slm train cpt | sft | dpo --config configs/<stage>.yaml [--dry-run] [--max-iters 100]
```

Synthetic SFT review (terminal accept/reject loop): `python scripts/review_sft.py`.
Merge + quantize a final adapter: `python scripts/merge_quantize.py`.

## Chat playground (`web/`)

A Next.js + [assistant-ui](https://github.com/assistant-ui/assistant-ui) front-end ships in `web/` for local dogfooding of the candidate model against task-specific system prompts (general, extraction, fact-check, summarize). It talks to whatever OpenAI-compatible runtime you've already wired up for the rest of the pipeline (`CIVIC_SLM_CANDIDATE_URL`), so no extra serving stack is needed.

```bash
pnpm --dir web install
pnpm --dir web dev    # http://localhost:3000
```

The dropdown defaults to **Gemma 4 (local)**; per-slot model strings are overridable via `CIVIC_SLM_GEMMA_MODEL`, `CIVIC_SLM_CIVIC_MODEL`, and `CIVIC_SLM_CANDIDATE_MODEL` so the UI's stable slugs map cleanly to whatever your server has loaded. The playground is for dogfooding only — production RAG/serving is out of scope (see [Out of scope](#out-of-scope) in CLAUDE.md).

## Eval-first

The training contract is **no training without a baseline**. The four benchmarks in `data/eval/` run against base Qwen 3.6 27B (`qwen3.6-27b-ud-mlx` in LM Studio) before any fine-tuning starts; those numbers are what every subsequent stage has to beat.

| Bench                   | What it measures                                     | Score                                                                                                                        |
| ----------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `civic_factuality`      | Q&A grounded in held-out docs                        | citation exact-match + answer similarity (`--similarity word_overlap` default; `bge` opt-in via the BGE dual-encoder cosine) |
| `refusal`               | refuses when context lacks the answer                | refusal recall + over-refusal precision (regex + mixed positive/negative class)                                              |
| `structured_extraction` | staff report → JSON                                  | field-level F1                                                                                                               |
| `side_by_side`          | open-ended U.S. municipal prompts vs base 7B and 72B | Claude or local-LLM judge w/ A/B position swap; runner fails fast if `$CIVIC_SLM_TEACHER_URL` isn't reachable                |

Current example counts (v0.2): **25 / 29 / 15 / 25** (multi-jurisdiction:
Austin TX, Houston TX, NYC, Phoenix AZ, Seattle WA, Cook County IL, Cuyahoga
County OH, Atlanta GA, Boston MA, Denver CO, Portland OR, plus the original
San-Clemente set). Target sizes for v1 per the training contract:
**200 / 100 / 50 / 100**.

### Run a baseline

The project targets **LM Studio** as the local inference runtime — see [docs/RUNTIMES.md](docs/RUNTIMES.md) for the full env table.

```bash
# 1. In LM Studio: download qwen3.6-27b-ud-mlx, Developer → Start Server (port 1234).
# 2. Source the project env block (points every CIVIC_SLM_* at LM Studio):
set -a; source .envrc.lmstudio; set +a

# 3. Sanity-check, then run a bench
uv run civic-slm doctor
uv run civic-slm eval run \
    --model base-qwen3.6-27b \
    --bench factuality \
    --bench-file data/eval/civic_factuality.jsonl
```

Reports land at `artifacts/evals/<model_id>/<bench>.{json,md}`.

## Baseline numbers (Qwen 2.5 7B, historical)

The v0 baselines (factuality 0.501, refusal 0.800, extraction 0.277) were
measured against Qwen 2.5 7B on a 10/14/5/10 bench under word-overlap, when
that was the project's base model. The project has since switched its base
to **Qwen 3.6 27B** (served via LM Studio); a re-baseline against the new
base lands in [#17](https://github.com/itsmeduncan/civic-slm/issues/17) and
populates `artifacts/evals/base-qwen3.6-27b/`. See `MODEL_CARD.md`
"Evaluation" for the live target table.

## Status

v0.1.0 shipped as an "infrastructure preview." v0.2.x has now landed all four
code-only tracks toward a v1 fine-tune: BGE dual-encoder factuality scorer
(opt-in via `--similarity bge`), training-pipeline robustness (`--smoke-test`,
`--resume`, signal-aware supervisor that flushes a checkpoint on Ctrl-C),
72B comparator wiring for `side_by_side`, and an eval scale-up to 25 / 29 /
15 / 25 across 11 U.S. jurisdictions. What's left for v0.2.0 is
maintainer-blocking — the san-clemente ToS audit (`docs/SOURCES.md`), the
first real crawl, the synth corpus, the actual CPT → SFT → DPO runs, and
the HF Hub publish.

See `CLAUDE.md` for the full project contract, `ARCHITECTURE.md` for design decisions, `docs/RUNTIMES.md` for picking a runtime, `docs/RECIPES.md` for adding new jurisdictions, `docs/GLOSSARY.md` for plain-language definitions of ML and civic terms, and `RELEASING.md` for the release checklist.
