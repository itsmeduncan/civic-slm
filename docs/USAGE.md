# How to use civic-slm, end to end

You're going to crawl real U.S. local-government documents, generate synthetic training data, fine-tune **Qwen 3.6 27B** (served locally as `qwen3.6-27b-ud-mlx` via LM Studio) on it, evaluate the result, and ship merged + quantized weights. The demo jurisdiction is San Clemente, CA; everything below works for any U.S. city, county, or township once you've added a recipe (see [RECIPES.md](RECIPES.md)). Every step writes a versioned artifact to disk. The whole thing runs on this Mac.

## Step 0 — One-time setup (5 minutes)

```bash
# From wherever you cloned the repo:
git clone https://github.com/itsmeduncan/civic-slm.git
cd civic-slm

# Install everything (core + ingest + synth + train + eval + dev)
uv sync --all-extras

# Browser-use needs a Chromium
uv run playwright install chromium

# Sanity check
uv run pytest                  # all tests should be green, ~0.15s
uv run civic-slm --help
```

Drop your secrets at `~/.config/civic-slm/.env`:

```
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
```

`HF_TOKEN` is for downloading models from Hugging Face faster (not strictly required). `ANTHROPIC_API_KEY` is for synth + the side-by-side judge + the browser-use crawler. `WANDB_API_KEY` is for training run logs.

**Run zero paid tokens?** Skip the Anthropic key, point synth/judge/crawler at a local teacher LLM (`CIVIC_SLM_LLM_BACKEND=local`), and lock the door with `CIVIC_SLM_STRICT_LOCAL=1`. Then `civic-slm doctor --strict-local` will exit non-zero if anything could reach a paid endpoint. Full setup in [RUNTIMES.md](RUNTIMES.md#strict-local-mode-zero-api-spend-with-proof).

### Going fully local (no Anthropic, no external APIs)

Synth, judge, and crawler all route through `civic_slm.llm.backend.select_backend()`, which picks based on env:

```bash
export CIVIC_SLM_LLM_BACKEND=local
export CIVIC_SLM_LM_STUDIO_URL=http://127.0.0.1:1234
export CIVIC_SLM_DEFAULT_MODEL=base-qwen3.6-27b      # registry label, see src/civic_slm/serve/models.py
```

Stand up a teacher model on port 8081 (Qwen2.5-72B-Instruct GGUF Q4 is the recommended choice — closest to Claude-quality synth). With ≥96GB unified memory, run candidate (7B-q4) and teacher (72B-q4) side by side. With less, swap one in at a time.

```bash
# teacher — uses ~40GB resident at Q4
# In LM Studio: load qwen3.6-27b-ud-mlx + your comparator on the same server (port 1234)

# candidate — uses ~5GB
# In LM Studio: load qwen3.6-27b-ud-mlx, then Developer → Start Server (port 1234)
```

`HF_TOKEN` and `WANDB_API_KEY` remain optional. Without `CIVIC_SLM_LLM_BACKEND=local`, behavior is unchanged (defaults to Anthropic for synth/judge/crawler).

## Train a model for your jurisdiction (the one-command path)

If your jurisdiction is already a registered recipe (`ls src/civic_slm/ingest/recipes/*.yaml`), you can skip steps 2-10 and let the composer drive the whole pipeline:

```bash
uv run civic-slm train jurisdiction santa-monica
```

What runs, in order, with each stage's status recorded to `artifacts/<slug>-pipeline/status.json` so a Ctrl-C + rerun picks up where you left off:

```
crawl → process → synth → prepare-cpt → prepare-sft
       → train cpt (200 iters) → fuse cpt
       → train sft (3 epochs)  → fuse v1
       → quantize (mlx-q4)     → eval (factuality / refusal / extraction)
```

Defaults (`--max-docs 50`, `--since 2024-01-01`, `--cpt-iters 200`, `--n-per-chunk 3`) are the Mac-128GB-validated knobs from PR #43. Tune via flags; `--dry-run` prints the planned stages without executing. `--skip-quantize` or `--skip-eval` short-circuits the tail when you only want the fused weights. Eval auto-passes `--allow-contamination` because the trained model has seen its own corpus — that's expected, not a bug, and the run config still records the overlap count for posterity.

Wall-clock for a San Clemente-sized corpus (~30 chunks → ~200 SFT pairs): ~3 hours plus the eval pass.

If your jurisdiction isn't a recipe yet, run `civic-slm new-recipe` first (see [RECIPES.md](RECIPES.md)). Want to query the model interactively after training? See [`civic-slm rag`](#step-12--ask-questions-about-your-jurisdiction) below.

The remaining sections (Steps 1-12) document each stage individually for the maintainer who wants to drive them by hand — useful for debugging, ablations, or running just one stage in isolation.

## Step 1 — Sanity-check the baseline (10 minutes, no GPU)

Confirm the eval harness still reproduces the committed numbers before you change anything. This is the floor every future stage has to clear.

Bring up LM Studio with the project's base model loaded:

1. Open LM Studio.
2. Search for and download `qwen3.6-27b-ud-mlx` (the candidate / base model).
3. Developer tab → **Start Server** (defaults to `http://127.0.0.1:1234`).

Then source the project env file so every `CIVIC_SLM_*` variable points at LM Studio:

```bash
set -a; source .envrc.lmstudio; set +a
```

Terminal 2 — sanity-check, then run all three available benches:

```bash
uv run civic-slm doctor                         # or: civic-slm doctor --strict-local for the zero-spend audit
```

```bash
for bench in factuality refusal extraction; do
  jsonl="data/eval/${bench}.jsonl"
  [ "$bench" = "factuality" ] && jsonl="data/eval/civic_factuality.jsonl"
  [ "$bench" = "extraction" ] && jsonl="data/eval/structured_extraction.jsonl"
  uv run civic-slm eval run \
      --model base-qwen2.5-7b \
      --bench "$bench" \
      --bench-file "$jsonl"
done
```

(`--model` is a registry label resolved through `src/civic_slm/serve/models.py` to BOTH the artifact directory and the served-model name — they cannot disagree. `--base-url` defaults to `$CIVIC_SLM_LM_STUDIO_URL`; pass it explicitly to override.)

Reports land at `artifacts/evals/base-qwen2.5-7b/{factuality,refusal,extraction}.{json,md}`. You should see roughly: factuality 0.501, refusal 0.800, extraction 0.277. If they drift, your harness changed — investigate before training.

## Step 2 — Crawl real documents (~15 minutes for 20 docs)

```bash
uv run civic-slm crawl san-clemente --since 2025-01-01 --max 20
```

What happens: a `browser-use` agent (LLM-driven; uses `ANTHROPIC_API_KEY` by default, or your local LLM if `CIVIC_SLM_LLM_BACKEND=local`) navigates the jurisdiction's website, finds council agendas, and returns structured `{title, meeting_date, source_url}` records. The harness fetches each PDF, sha256-deduplicates against `data/raw/manifest.jsonl`, extracts text via `pypdf`, and appends to the manifest. Re-running is idempotent — only new docs land.

Verify:

```bash
wc -l data/raw/manifest.jsonl
ls -la data/raw/san-clemente/
```

To add another jurisdiction (any U.S. city, county, township, school district): run `civic-slm new-recipe`, which prompts for slug / state / vendor / start URL and writes a YAML stub under `src/civic_slm/ingest/recipes/`. The crawler auto-discovers it — no `_RECIPES` dict to edit. Full walkthrough in [RECIPES.md](RECIPES.md).

## Step 2.5 — Crawl meeting videos (optional)

Council meeting recordings are where the actual deliberation lives. To pull them, give your recipe a `discover_videos` method that returns a list of `DiscoveredVideo` (the [RECIPES.md](RECIPES.md) "Adding video sources" section walks through this), then:

```bash
uv run civic-slm crawl-videos san-clemente --since 2025-01-01 --max 20
```

Per video: `yt-dlp` downloads `bestaudio.m4a` + any human SRT/VTT + the YouTube auto-caption track. `civic_slm.ingest.video.transcript` walks a priority chain — human SRT/VTT → YouTube auto-caption → Whisper ASR fallback (`mlx-whisper`, lazy-imported, Apple Silicon only). The transcript text lands in the same `data/raw/manifest.jsonl` as a `meeting_transcript` doc, indistinguishable from any PDF downstream.

ASR runs at ~1× real-time on M-series, so a 3-hour meeting takes ~3 hours of compute. Most public-meeting channels publish auto-captions, so the Whisper path is rare in practice.

## Step 3 — Chunk the corpus into training-ready pieces

```bash
civic-slm process san-clemente
# → reads manifest entries for this jurisdiction
# → extracts text from each PDF under data/raw/
# → writes data/processed/san-clemente.jsonl
```

Chunker emits 1024-token chunks with 128-token overlap, tracking ALL-CAPS and numbered headings as `section_path`. Missing files in the manifest are logged and skipped, not fatal — so a partially-completed crawl can still be processed.

## Step 4 — Generate synthetic SFT pairs (~30 minutes for 5k examples, costs ~$5–15 in Anthropic credits)

The pipeline takes each chunk, hands it to Claude Opus 4.7 with a per-task prompt template, and gets back `{system, input, output}` triples. Every line is Pydantic-validated; invalid lines are dropped and logged.

```bash
civic-slm synth san-clemente
# → reads data/processed/san-clemente.jsonl
# → resolves state + dominant doc_type from the manifest
# → writes data/sft/san-clemente.jsonl (resumable; rerun is a no-op once complete)
```

Useful flags:

```bash
civic-slm synth san-clemente \
  --n-per-chunk 3 \
  --rounds 12 \
  --concurrency 4 \
  --task qa_grounded --task refusal \
  --doc-type agenda \
  --out data/sft/san-clemente-v0.jsonl
```

`--rounds K` runs the full chunk × task sweep K times, stamping each pass with an incrementing `synth_round` in provenance. Resume keys on `(chunk, task, round)`, so re-running `--rounds 4` against a file that already contains rounds 0–1 generates rounds 2–5 (and skips 0–1 for free). To scale a ~35-chunk corpus to **~5,000 examples** keep `--n-per-chunk 3` and bump `--rounds 12` (35 × 4 × 3 × 12 ≈ 5,040). For one-command training, `civic-slm train jurisdiction <slug> --synth-rounds 12` threads the flag through.

If `Step 3` hasn't run yet, the CLI exits early with a clear pointer to `civic-slm process {jurisdiction}`.

The four task templates (`qa_grounded`, `refusal`, `extract`, `summarize`) live at `src/civic_slm/synth/prompts/*.md`. Each prompt's SHA is hashed into `Provenance.prompt_sha` on every example, so when you tweak a template, you can re-generate just the stale ones.

## Step 5 — Human-curate the first ~500 (1-2 hours)

This catches systemic problems (leading questions, ungrounded answers, repetitive patterns) that schema validation can't see. **Do this once, before scaling up the synth run.**

```bash
uv run civic-slm review-sft san-clemente --limit 500
```

Press `a` to accept, `r` to reject, `s` to skip-for-now, `q` to quit. Progress is saved in `data/sft/.review_state.json` so you can resume across sessions. Accepts land in `data/sft/san-clemente.curated.jsonl`.

Once you've curated, materialize the train/valid SFT splits and the CPT corpus:

```bash
uv run civic-slm prepare-sft data/sft/san-clemente.curated.jsonl
# → data/sft/san-clemente.train.jsonl + data/sft/san-clemente.valid.jsonl  (chat format)
uv run civic-slm prepare-cpt san-clemente
# → data/processed/cpt.jsonl  (one {"text": ...} per line, mlx_lm text-mode)
```

## Step 6 — CPT smoke run (10 minutes)

Continued pretraining surfaces the model to civic vocabulary before any task-specific tuning. Always do the 100-step smoke run before committing to a full train.

```bash
# See the command without running it
uv run civic-slm train cpt --smoke-test --dry-run

# Actually run 100 steps
uv run civic-slm train cpt --smoke-test
```

Watch the loss in the terminal output. You want to see it dropping monotonically. If memory pressure spikes, drop `batch_size` in `configs/cpt.yaml` and add gradient accumulation. If loss is flat, prompt format or LR is wrong.

When the smoke run looks healthy, commit to the real run:

```bash
uv run civic-slm train cpt   # uses configs/cpt.yaml: 2000 iters, LR 1e-5, LoRA r=64
```

Adapter lands at `artifacts/qwen-civic-cpt/`. The trainer refuses to overwrite an existing adapter without `--resume`; if you want to extend a prior run, pass `--resume`. Ctrl-C is now safe — the supervisor propagates SIGINT so the child flushes a checkpoint before exiting.

## Step 7 — SFT (1-3 hours)

```bash
uv run civic-slm train sft --smoke-test            # 50-step smoke
uv run civic-slm train sft                         # real run (3 epochs)
```

This trains on `data/sft/v0.curated.jsonl` over 3 epochs at LR 2e-4, LoRA r=32 α=64, packing on. Adapter lands at `artifacts/qwen-civic-sft/`.

**Re-run all baselines now.** Adapter inference: serve via MLX with the adapter path, then point eval at it. The gate to DPO is "beats base on ≥3/4 benches."

## Step 8 — DPO (optional for v0)

You need preference pairs first. Cheapest source: run the SFT model to generate two completions per prompt, judge them with Claude, take the chosen/rejected. Drop pairs at `data/dpo/v0.jsonl` (`PreferencePair` schema).

```bash
uv run civic-slm train dpo --smoke-test            # 50-step smoke
uv run civic-slm train dpo                         # real run (1 epoch)
```

If `mlx_lm.dpo` errors, ship v0 as CPT+SFT only and add DPO in v1. CLAUDE.md explicitly allows that fallback.

## Step 9 — Merge + quantize for release

```bash
uv run civic-slm merge \
    --adapter-dir artifacts/qwen-civic-sft \
    --base-model qwen3.6-27b-ud-mlx \
    --version v1
```

Outputs:

- `artifacts/qwen-civic-v1-mlx-q4/` — primary Mac artifact, runs in LM Studio.
- `artifacts/qwen-civic-v1-gguf-q5km/qwen-civic-v1-q5_k_m.gguf` — for Ollama / llama.cpp users.

GGUF requires `brew install llama.cpp` first. If you don't need it: `--skip-gguf`.

## Step 10 — Final eval against the released artifact

```bash
# terminal 1 — any runtime can serve the fused artifact
# In LM Studio: load qwen3.6-27b-ud-mlx, then Developer → Start Server (port 1234)
# OR: ollama create qwen-civic-v1 -f Modelfile  # then `ollama run qwen-civic-v1`
# OR: import the GGUF into LM Studio and reload the server

# terminal 2 — point eval at whichever you started.
# The civic-slm-v1 label resolves to whichever served name LM Studio reports
# for your fused artifact (see src/civic_slm/serve/models.py — bump that one
# string when v1 ships).
for bench in factuality refusal extraction; do
  jsonl="data/eval/${bench}.jsonl"
  [ "$bench" = "factuality" ] && jsonl="data/eval/civic_factuality.jsonl"
  [ "$bench" = "extraction" ] && jsonl="data/eval/structured_extraction.jsonl"
  uv run civic-slm eval run --model civic-slm-v1 --bench "$bench" --bench-file "$jsonl"
done
```

For `side_by_side`, also have the comparator loaded in LM Studio (the default `comparator-gemma-4-31b` resolves to `gemma-4-31b-it-mlx`).

```bash
uv run civic-slm eval side-by-side \
    --candidate civic-slm-v1 \
    --comparator comparator-gemma-4-31b
```

The judge runs each comparison twice (A/B swapped); only agreement counts. Score is win-rate (1.0 / 0.5 / 0.0).

## Step 11 — Tag and ship

```bash
# Bump VERSION
echo "0.1.0" > VERSION

# Add a CHANGELOG entry under [Unreleased] with the new eval numbers
# (use the [0.0.1] entry as a template)

git add VERSION CHANGELOG.md artifacts/evals/qwen-civic-v1/
git commit -m "feat: ship v0.1.0 fine-tune"
git tag -a v0.1.0 -m "v0.1.0: first fine-tune"
```

Push weights to HF Hub when you're ready:

```bash
uv run huggingface-cli upload itsmeduncan/qwen-civic-v1-mlx-q4 artifacts/qwen-civic-v1-mlx-q4
uv run huggingface-cli upload itsmeduncan/qwen-civic-v1-gguf-q5km artifacts/qwen-civic-v1-gguf-q5km
```

## Step 12 — Dogfood in the chat playground (optional)

`web/` is a Next.js app built on [assistant-ui](https://github.com/assistant-ui/assistant-ui) for poking at the candidate model interactively. It talks to the same `CIVIC_SLM_LM_STUDIO_URL` your eval harness uses, so no extra serving stack is needed.

The dropdown's three slots map to registry labels via `web/src/lib/models.ts` (must mirror `src/civic_slm/serve/models.py`). To change which Civic SLM build the UI points at, bump the `civic-slm-v1` row in both files.

```bash
# 1. Make sure LM Studio is up with the relevant models loaded (see docs/RUNTIMES.md).
pnpm --dir web install
pnpm --dir web dev    # http://localhost:3000
```

The sidebar swaps system prompts across four task presets (general, extraction, fact-check, summarize) without leaving the thread. The dropdown exposes the comparator Gemma slot, the trained Civic SLM slot, and base Qwen for side-by-side prompt sniffing. This UI is for development feedback only — production multi-tenant serving is out of scope (see [CLAUDE.md "Out of scope"](../CLAUDE.md)).

## Step 13 — Ask questions about your jurisdiction (local RAG)

Once you've trained a per-jurisdiction model (Step 0's `civic-slm train jurisdiction` or the manual chain), build a retrieval index over the same processed chunks and query the model with citations:

```bash
# One-time: index your jurisdiction's chunks against BGE embeddings.
uv run civic-slm rag index santa-monica

# Start mlx_lm.server pointed at the trained model:
uv run mlx_lm.server --model artifacts/santa-monica-v1-fused \
  --chat-template-args '{"enable_thinking": false}' \
  --port 1234
```

In another shell, ask:

```bash
uv run civic-slm rag ask santa-monica \
  "When did the council last discuss the Pier reconstruction project?"
# → grounded answer + [N] citations linking back to source PDFs in the manifest
```

Or run the OpenAI-compatible RAG shim and point the playground at it:

```bash
uv run civic-slm rag serve santa-monica --port 8767
# Then start web/ with CIVIC_SLM_LM_STUDIO_URL=http://127.0.0.1:8767
pnpm --dir web dev
```

Scope reminder: this RAG path is for local single-jurisdiction dogfooding. It binds to `127.0.0.1` only, has no auth, no persistence, and no multi-tenant authorization. Production-grade RAG belongs in a separate project (CLAUDE.md "Out of scope").

## Day-to-day commands (cheat sheet)

```bash
# Tests
uv run pytest

# Full quality check
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest

# Crawl
uv run civic-slm crawl san-clemente --max 20

# Eval against any served model
uv run civic-slm eval run --model <id> --bench factuality \
    --bench-file data/eval/civic_factuality.jsonl --base-url http://127.0.0.1:1234

# Train (with smoke first)
uv run civic-slm train cpt --smoke-test
uv run civic-slm train cpt

# SFT review
uv run civic-slm review-sft

# Release
uv run civic-slm merge --adapter-dir artifacts/qwen-civic-sft \
    --base-model qwen3.6-27b-ud-mlx --version v1
```

## What to watch for

- **Eval harness drift.** If you change a scorer, re-run the baseline before training. A scorer change masks model regressions.
- **Synth quality.** Curate the first 500 by hand. If you see patterns (the model always answers in the same shape, refuses too aggressively, hallucinates dates), the prompt template needs tuning. Rotate the `prompt_sha` and re-generate.
- **Memory pressure during training.** Drop batch size first, then increase grad accumulation, before reducing LoRA rank.
- **Ground truth in extraction.** The base model nests under `staff_report` instead of producing flat JSON. Make sure your synth examples enforce the flat schema explicitly, or you'll bake the wrong shape in.

That's the whole loop.
