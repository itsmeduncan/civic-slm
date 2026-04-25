# Architecture

This is a single-Mac, eval-first pipeline that turns crawled **U.S. local-government** documents (cities, counties, townships, school districts) into a domain-specialized fine-tune of Qwen2.5-7B-Instruct. Every stage between crawl and release writes a versioned, schema-validated artifact to disk; nothing is in-memory-only, and re-runs are idempotent. San Clemente, CA ships as the demo recipe; the recipe pattern (`src/civic_slm/ingest/recipes/_template.py`) generalizes to any U.S. jurisdiction.

## Pipeline

```
                            (Apple Silicon, single Mac)

  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  city sites          Anthropic API           MLX-LM            HF Hub  │
  │      │                    │                    │                  ▲    │
  │      ▼                    ▼                    ▼                  │    │
  │  ┌──────┐  PDFs   ┌──────────┐ JSONL    ┌────────┐  weights  ┌────────┤
  │  │crawl │────────▶│  synth   │─────────▶│ train  │──────────▶│ merge  │
  │  └──────┘         └──────────┘          │CPT/SFT/│           │   +    │
  │      │  manifest        │               │  DPO   │           │quant.  │
  │      ▼                  ▼               └────────┘           └────────┤
  │  data/raw/         data/sft/v0.jsonl        │                   │ │   │
  │      │                  │                   ▼                   │ │   │
  │      ▼                  ▼          artifacts/qwen-civic-*       │ │   │
  │   chunk ──────────► DocumentChunks                              │ │   │
  │                          │                                      ▼ ▼   │
  │                          └─────────────────────► eval ◀───── MLX-q4   │
  │                                                    │         GGUF Q5_K_M
  │                              4 benches: factuality, refusal,           │
  │                              extraction, side_by_side                  │
  │                                       │                                │
  │                                       ▼                                │
  │                          artifacts/evals/<model>/<bench>.json          │
  └────────────────────────────────────────────────────────────────────────┘
```

Every stage logs to W&B project `civic-slm`, run name `{stage}-{git_sha}-{timestamp}`. Intermediate artifacts are gitignored under `artifacts/`; only the manifest, eval JSONLs, and reports are committed.

## Why a single Mac

CLAUDE.md originally specified Ubuntu + RTX 4090. We swapped to Apple Silicon for two reasons: (1) the dev machine is a Mac, (2) the model size + LoRA rank fit comfortably in unified memory at 4-bit. Doing it on the Mac forces discipline on batch sizes and rules out a class of "just throw more GPUs at it" decisions.

| Original (CUDA)    | Apple Silicon equivalent       | Why                                                                                       |
| ------------------ | ------------------------------ | ----------------------------------------------------------------------------------------- |
| Unsloth (training) | MLX-LM `mlx-lm.lora`           | First-party Apple LoRA/QLoRA support; no Mac alternative                                  |
| vLLM (serving)     | **any OpenAI-compatible HTTP** | We don't lock you in: MLX, Ollama, LM Studio, llama.cpp, vLLM (Linux), or your own server |
| AWQ-4bit (release) | MLX 4-bit + GGUF Q5_K_M        | AWQ is CUDA-only; both released so MLX users _and_ Ollama/LM Studio/llama.cpp users run   |
| TRL DPO            | `mlx_lm.dpo`                   | Newer, but in-tree                                                                        |

**Inference is vendor-agnostic; training is MLX.** Anyone can run the released weights on whatever runtime they prefer; you only need an Apple Silicon Mac with MLX-LM if you want to _retrain_. See `docs/RUNTIMES.md` for serving setup per runtime.

## Why an LLM-driven crawler

U.S. local-government sites run on a long tail of vendor platforms: Granicus, Legistar, CivicPlus, IQM, PrimeGov, Municode, plus countless custom WordPress and Drupal builds. Hand-written CSS-selector scrapers per platform rot fast and don't generalize across the 19,000+ U.S. municipalities. We hand a `browser-use` agent a natural-language instruction (`"find council agendas for the last 12 months"`) and let it navigate — same recipe shape works whether the jurisdiction is a CA city on Granicus, a TX county on CivicPlus, or a NY township on a custom CMS. We pay for it in API tokens and wallclock; at the v0 corpus scale (thousands of docs, not millions) the tradeoff is right.

Recipe surface: `src/civic_slm/ingest/recipes/<jurisdiction>.py`. Each recipe exposes `jurisdiction` (kebab-case slug) + `state` (2-letter postal code) and an async `discover(since, max_docs)` returning `DiscoveredDoc` records. The `crawl()` orchestrator handles fetching, deduping by sha256 against `data/raw/manifest.jsonl`, extracting text (PDF via `pypdf`), and appending to the manifest. Re-runs are idempotent. Adding a new jurisdiction is a copy-paste of `_template.py` — see `docs/RECIPES.md`.

## Why eval-first

The training contract is: **don't train until the eval harness reproduces a baseline against the base model**. `data/eval/` holds four hand-curated, schema-validated benchmark JSONLs. We run them against base Qwen2.5-7B-Instruct (MLX 4-bit) to commit a baseline before any fine-tuning. Gating training stages on "≥3/4 benches improve over the prior version" is the only way to know our changes are working — we lock down the floor before we start moving the ceiling.

| Bench        | Question shape                  | Scorer                                                                  |
| ------------ | ------------------------------- | ----------------------------------------------------------------------- |
| factuality   | grounded Q&A from held-out docs | citation exact-match + word-overlap (BGE reranker swap planned)         |
| refusal      | adversarial Qs not answerable   | regex over canonical refusal phrases (Claude judge for ambiguous cases) |
| extraction   | doc → JSON                      | field-level F1 over flat dicts                                          |
| side_by_side | open-ended civic prompt         | Claude Sonnet 4.6 pairwise judge with A/B position swap                 |

`side_by_side` runs the candidate against a comparator (currently Qwen2.5-7B; Qwen2.5-72B GGUF Q4 once it's installed) and only counts a "win" if the judge agrees in both A→B and B→A orderings. This is the cheapest defense against position bias.

## Data contracts

Every artifact between stages is a Pydantic v2 model in `src/civic_slm/schema.py`:

| Model                | Crosses                                     | Key fields                                                                     |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| `CivicDocument`      | crawl → chunker                             | id, city, doc_type, source_url, sha256, raw_path, text                         |
| `DocumentChunk`      | chunker → synth, training                   | doc_id, chunk_idx, text, token_count, section_path                             |
| `Provenance`         | synth → SFT                                 | generator, model, prompt_sha, created_at                                       |
| `InstructionExample` | synth → SFT (`data/sft/`)                   | task, system, input, output, source_chunk_ids, provenance                      |
| `PreferencePair`     | judge → DPO (`data/dpo/`)                   | prompt, chosen, rejected, rationale                                            |
| `EvalExample`        | benches → eval runner (discriminated union) | bench-specific shape per `factuality \| refusal \| extraction \| side_by_side` |
| `EvalResult`         | eval runner → reports                       | model_id, bench, example_id, prediction, score, judge_notes, latency_ms        |

All models are `frozen=True, extra="forbid"`. JSON round-trip is tested per model. If it doesn't validate, it doesn't land.

## Repository layout

```
civic-slm/
├── pyproject.toml         # uv-managed, grouped extras: ingest/synth/train/eval
├── ruff.toml              # 100 cols, py311 target
├── pyrightconfig.json     # strict mode
├── VERSION                # source of truth for releases
├── CHANGELOG.md           # Keep-a-Changelog
├── CLAUDE.md              # project contract for Claude Code
├── README.md              # user-facing entry point
├── ARCHITECTURE.md        # this file
├── CONTRIBUTING.md        # dev setup + workflow
├── LICENSE                # MIT
│
├── docs/
│   ├── USAGE.md           # end-to-end walkthrough
│   ├── RECIPES.md         # add a new U.S. jurisdiction
│   └── RUNTIMES.md        # serve via MLX / Ollama / LM Studio / llama.cpp
│
├── configs/               # cpt.yaml, sft.yaml, dpo.yaml — full training contract
├── data/
│   ├── raw/               # crawled bytes (gitignored except manifest.jsonl)
│   ├── processed/         # cleaned, chunked (gitignored)
│   ├── sft/               # InstructionExample JSONLs
│   ├── dpo/               # PreferencePair JSONLs
│   └── eval/              # 4 benchmark JSONLs (committed)
│
├── src/civic_slm/
│   ├── schema.py          # Pydantic data contracts
│   ├── config.py          # ~/.config/civic-slm/.env loader, require()
│   ├── logging.py         # structlog setup (JSON in non-TTY, pretty in TTY)
│   ├── cli.py             # umbrella Typer: crawl, doctor, eval, train, version
│   ├── doctor.py          # `civic-slm doctor` — env + runtime sanity check
│   ├── llm/               # Backend abstraction (anthropic | local OpenAI-compatible)
│   ├── ingest/            # PDF + video crawlers, recipes, chunker
│   │   ├── recipes/       # _template, _youtube, _browser helpers + per-jurisdiction
│   │   └── video/         # caption (VTT/SRT), youtube (yt-dlp), transcript, asr (whisper)
│   ├── synth/             # backend-agnostic synth generator, taxonomy prompts as .md
│   ├── train/             # MLX trainer wrappers (cpt, sft, dpo, common, dataset)
│   ├── eval/              # runner, scorers, judge, side_by_side runner
│   └── serve/             # ChatClient + Runtime presets + env-driven defaults
│
├── scripts/
│   ├── merge_quantize.py  # fuse adapter, MLX-q4 + GGUF Q5_K_M export
│   └── review_sft.py      # terminal accept/reject CLI for synth review
│
└── tests/                 # pytest, fast, no GPU required
```

## CLI shape

The `civic-slm` umbrella registers each stage's leaf function directly (not as a sub-app), so subcommands are flat:

```
civic-slm doctor                                              # sanity-check env + runtime
civic-slm crawl              --jurisdiction <slug>  [--since ISO]  [--max N]
civic-slm crawl-videos       --jurisdiction <slug>  [--since ISO]  [--max N]   # YT + ASR
civic-slm eval run           --model <id>  --bench <name>  --bench-file <path>
civic-slm eval side-by-side  --candidate-model <id>  [--candidate-url ...]  [--comparator-url ...]
civic-slm train cpt          --config configs/cpt.yaml  [--dry-run]  [--max-iters N]
civic-slm train sft          --config configs/sft.yaml  [--dry-run]  [--max-iters N]
civic-slm train dpo          --config configs/dpo.yaml  [--dry-run]  [--max-iters N]
civic-slm version
```

`eval run` and `eval side-by-side` default `--base-url` / `--served-model` from `CIVIC_SLM_CANDIDATE_URL` / `CIVIC_SLM_CANDIDATE_MODEL` (and `_TEACHER_*` for the comparator). See `docs/RUNTIMES.md`.

Two scripts live outside the umbrella because they're rare and one-shot:

- `python scripts/review_sft.py` — terminal accept/reject loop for the first ~500 synthetic examples; persists progress in `data/sft/.review_state.json`.
- `python scripts/merge_quantize.py` — fuse + quantize a final adapter for release.

## Design decisions worth remembering

- **No LangChain.** Synth, the judge, and the crawler all route through `civic_slm.llm.backend.select_backend()`, which picks Anthropic SDK or a local OpenAI-compatible endpoint based on `CIVIC_SLM_LLM_BACKEND`. Prompts are `.md` files in `src/civic_slm/synth/prompts/`, hashed into `Provenance.prompt_sha` so we can re-generate only stale examples when a template changes.
- **Lazy imports for optional deps.** `pypdf`, `browser_use`, `anthropic`, `mlx_lm`, `wandb` are all lazy-imported at use sites so the core package + tests work without GPU/heavy deps installed.
- **Trainer wrappers shell out to MLX-LM CLIs** instead of importing them. Re-implementing MLX-LM's training loop is more fragile than delegating; the cost is no per-step W&B hooks.
- **Scoring stays cheap by default.** Factuality uses jaccard word-overlap as a stand-in; the BGE reranker swap is a `similarity_fn` parameter on `score_factuality`. Same trick for the refusal classifier.
- **Position-bias mitigation.** `judge_with_position_swap` runs the pairwise judge twice with A/B swapped; only agreement counts. Anything else is a tie.
- **Append-only manifest with sha256 dedupe.** Crawls are idempotent. The manifest is the audit trail even if a city later changes a URL.
- **Caption-first transcripts.** Video ingestion prefers human SRT/VTT, then YouTube auto-captions, only falling back to Whisper ASR. Caption parsing handles YouTube's "rolling cue" pattern (each cue extends the previous) by collapsing same-speaker prefix-matching cues. Real diarization is a v1 line item; v0 preserves speaker labels heuristically when the source provides them.

## What's deliberately out of scope

- RAG serving (separate project; this is the model).
- Frontend / UI.
- Multi-machine training (single-Mac constraint forces discipline).
- AWQ. AWQ is CUDA-only; on Apple Silicon, MLX-q4 is the native equivalent.
