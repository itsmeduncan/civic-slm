# How to use civic-slm, end to end

You're going to crawl real U.S. local-government documents, generate synthetic training data, fine-tune Qwen2.5-7B on it, evaluate the result, and ship merged + quantized weights. The demo jurisdiction is San Clemente, CA; everything below works for any U.S. city, county, or township once you've added a recipe (see [RECIPES.md](RECIPES.md)). Every step writes a versioned artifact to disk. The whole thing runs on this Mac.

## Step 0 — One-time setup (5 minutes)

```bash
cd ~/Projects/src/github.com/itsmeduncan/civic-slm

# Install everything (core + ingest + synth + train + eval + dev)
uv sync --all-extras

# Browser-use needs a Chromium
uv run playwright install chromium

# Sanity check
uv run pytest                  # 37 tests, ~0.1s, all green
uv run civic-slm --help
```

Drop your secrets at `~/.config/civic-slm/.env`:

```
HF_TOKEN=hf_...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
```

`HF_TOKEN` is for downloading models from Hugging Face faster (not strictly required). `ANTHROPIC_API_KEY` is for synth + the side-by-side judge + the browser-use crawler. `WANDB_API_KEY` is for training run logs.

### Going fully local (no Anthropic, no external APIs)

Synth, judge, and crawler all route through `civic_slm.llm.backend.select_backend()`, which picks based on env:

```bash
export CIVIC_SLM_LLM_BACKEND=local
export CIVIC_SLM_LOCAL_LLM_URL=http://127.0.0.1:8081
export CIVIC_SLM_LOCAL_LLM_MODEL=default              # whatever the local server reports
```

Stand up a teacher model on port 8081 (Qwen2.5-72B-Instruct GGUF Q4 is the recommended choice — closest to Claude-quality synth). With ≥96GB unified memory, run candidate (7B-q4) and teacher (72B-q4) side by side. With less, swap one in at a time.

```bash
# teacher — uses ~40GB resident at Q4
llama-server -m ~/models/qwen2.5-72b-instruct-q4_k_m.gguf -c 8192 --port 8081

# candidate — uses ~5GB
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

`HF_TOKEN` and `WANDB_API_KEY` remain optional. Without `CIVIC_SLM_LLM_BACKEND=local`, behavior is unchanged (defaults to Anthropic for synth/judge/crawler).

## Step 1 — Sanity-check the baseline (10 minutes, no GPU)

Confirm the eval harness still reproduces the committed numbers before you change anything. This is the floor every future stage has to clear.

Terminal 1 — start an MLX server with the base model:

```bash
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

First run downloads ~4.5GB. Wait until you see `Starting httpd at http://127.0.0.1:8080`.

Terminal 2 — run all three available benches:

```bash
for bench in factuality refusal extraction; do
  jsonl="data/eval/${bench}.jsonl"
  [ "$bench" = "factuality" ] && jsonl="data/eval/civic_factuality.jsonl"
  [ "$bench" = "extraction" ] && jsonl="data/eval/structured_extraction.jsonl"
  uv run civic-slm eval run \
      --model base-qwen2.5-7b \
      --bench "$bench" \
      --bench-file "$jsonl" \
      --base-url http://127.0.0.1:8080 \
      --served-model mlx-community/Qwen2.5-7B-Instruct-4bit
done
```

Reports land at `artifacts/evals/base-qwen2.5-7b/{factuality,refusal,extraction}.{json,md}`. You should see roughly: factuality 0.501, refusal 0.800, extraction 0.277. If they drift, your harness changed — investigate before training.

## Step 2 — Crawl real documents (~15 minutes for 20 docs)

```bash
uv run civic-slm crawl --jurisdiction san-clemente --since 2025-01-01 --max 20
```

What happens: a `browser-use` agent (LLM-driven; uses `ANTHROPIC_API_KEY` by default, or your local LLM if `CIVIC_SLM_LLM_BACKEND=local`) navigates the jurisdiction's website, finds council agendas, and returns structured `{title, meeting_date, source_url}` records. The harness fetches each PDF, sha256-deduplicates against `data/raw/manifest.jsonl`, extracts text via `pypdf`, and appends to the manifest. Re-running is idempotent — only new docs land.

Verify:

```bash
wc -l data/raw/manifest.jsonl
ls -la data/raw/san-clemente/
```

To add another jurisdiction (any U.S. city, county, township, school district), copy `src/civic_slm/ingest/recipes/_template.py` to `<jurisdiction>.py`, edit three things (slug, state, instruction), and register it in `src/civic_slm/ingest/crawl.py`'s `_RECIPES` dict. Full walkthrough in [RECIPES.md](RECIPES.md).

## Step 3 — Chunk the corpus into training-ready pieces

Right now chunking happens lazily inside synth. If you want a clean offline chunk pass:

```bash
uv run python -c "
from pathlib import Path
from civic_slm.ingest import manifest
from civic_slm.ingest.pdf import chunk_text

docs = manifest.load_manifest(Path('data'))
out = Path('data/processed/chunks.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as fh:
    for doc in docs:
        for chunk in chunk_text(doc.id, doc.text):
            fh.write(chunk.model_dump_json() + '\n')
print(f'wrote {out}')
"
```

Chunker emits 1024-token chunks with 128-token overlap, tracking ALL-CAPS and numbered headings as `section_path`.

## Step 4 — Generate synthetic SFT pairs (~30 minutes for 5k examples, costs ~$5–15 in Anthropic credits)

The pipeline takes each chunk, hands it to Claude Opus 4.7 with a per-task prompt template, and gets back `{system, input, output}` triples. Every line is Pydantic-validated; invalid lines are dropped and logged.

```bash
uv run python -c "
import asyncio
from pathlib import Path
from civic_slm.ingest import manifest
from civic_slm.ingest.pdf import chunk_text
from civic_slm.synth.generate import generate_corpus

async def go():
    docs = manifest.load_manifest(Path('data'))
    chunks = [c for d in docs for c in chunk_text(d.id, d.text)]
    n = await generate_corpus(
        chunks=chunks,
        jurisdiction='san-clemente',
        state='CA',
        doc_type='agenda',
        out_path=Path('data/sft/v0.jsonl'),
        n_per_chunk=3,            # 3 examples per chunk per task
        concurrency=4,            # 4 simultaneous teacher-LLM calls
    )
    print(f'wrote {n} examples')

asyncio.run(go())
"
```

The four task templates (`qa_grounded`, `refusal`, `extract`, `summarize`) live at `src/civic_slm/synth/prompts/*.md`. Each prompt's SHA is hashed into `Provenance.prompt_sha` on every example, so when you tweak a template, you can re-generate just the stale ones.

## Step 5 — Human-curate the first ~500 (1-2 hours)

This catches systemic problems (leading questions, ungrounded answers, repetitive patterns) that schema validation can't see. **Do this once, before scaling up the synth run.**

```bash
uv run python scripts/review_sft.py --input-path data/sft/v0.jsonl --limit 500
```

Press `a` to accept, `r` to reject, `s` to skip-for-now, `q` to quit. Progress is saved in `data/sft/.review_state.json` so you can resume across sessions. Accepts land in `data/sft/v0.curated.jsonl`.

## Step 6 — CPT smoke run (10 minutes)

Continued pretraining surfaces the model to civic vocabulary before any task-specific tuning. Always do the 100-step smoke run before committing to a full train.

```bash
# See the command without running it
uv run civic-slm train cpt --max-iters 100 --dry-run

# Actually run 100 steps
uv run civic-slm train cpt --max-iters 100
```

Watch the loss in the terminal output. You want to see it dropping monotonically. If memory pressure spikes, drop `batch_size` in `configs/cpt.yaml` and add gradient accumulation. If loss is flat, prompt format or LR is wrong.

When the smoke run looks healthy, commit to the real run:

```bash
uv run civic-slm train cpt   # uses configs/cpt.yaml: 2000 iters, LR 1e-5, LoRA r=64
```

Adapter lands at `artifacts/qwen-civic-cpt/`.

## Step 7 — SFT (1-3 hours)

```bash
uv run civic-slm train sft --max-iters 100        # smoke
uv run civic-slm train sft                         # real run
```

This trains on `data/sft/v0.curated.jsonl` over 3 epochs at LR 2e-4, LoRA r=32 α=64, packing on. Adapter lands at `artifacts/qwen-civic-sft/`.

**Re-run all baselines now.** Adapter inference: serve via MLX with the adapter path, then point eval at it. The gate to DPO is "beats base on ≥3/4 benches."

## Step 8 — DPO (optional for v0)

You need preference pairs first. Cheapest source: run the SFT model to generate two completions per prompt, judge them with Claude, take the chosen/rejected. Drop pairs at `data/dpo/v0.jsonl` (`PreferencePair` schema).

```bash
uv run civic-slm train dpo --max-iters 50          # smoke
uv run civic-slm train dpo                         # real run
```

If `mlx_lm.dpo` errors, ship v0 as CPT+SFT only and add DPO in v1. CLAUDE.md explicitly allows that fallback.

## Step 9 — Merge + quantize for release

```bash
uv run python scripts/merge_quantize.py \
    --adapter-dir artifacts/qwen-civic-sft \
    --base-model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --version v1
```

Outputs:

- `artifacts/qwen-civic-v1-mlx-q4/` — primary Mac artifact, runs in `mlx_lm.server`.
- `artifacts/qwen-civic-v1-gguf-q5km/qwen-civic-v1-q5_k_m.gguf` — for Ollama / llama.cpp users.

GGUF requires `brew install llama.cpp` first. If you don't need it: `--skip-gguf`.

## Step 10 — Final eval against the released artifact

```bash
# terminal 1
uv run mlx_lm.server --model artifacts/qwen-civic-v1-mlx-q4 --port 8080

# terminal 2 — same eval commands as Step 1, but with --model qwen-civic-v1
for bench in factuality refusal extraction; do
  jsonl="data/eval/${bench}.jsonl"
  [ "$bench" = "factuality" ] && jsonl="data/eval/civic_factuality.jsonl"
  [ "$bench" = "extraction" ] && jsonl="data/eval/structured_extraction.jsonl"
  uv run civic-slm eval run \
      --model qwen-civic-v1 --bench "$bench" --bench-file "$jsonl" \
      --base-url http://127.0.0.1:8080 \
      --served-model artifacts/qwen-civic-v1-mlx-q4
done
```

For `side_by_side`, you also need a comparator on port 8081. The plan calls for Qwen2.5-72B GGUF Q4 via `llama-server` — only run this if your Mac has ≥64GB unified memory.

```bash
# terminal 3 — comparator (only if you have the GGUF weights)
llama-server -m ~/models/qwen2.5-72b-instruct-q4_k_m.gguf -c 8192 --port 8081

# terminal 2
uv run civic-slm eval side-by-side \
    --candidate-model qwen-civic-v1 \
    --candidate-url http://127.0.0.1:8080 \
    --candidate-served artifacts/qwen-civic-v1-mlx-q4 \
    --comparator-url http://127.0.0.1:8081 \
    --comparator-served default
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

## Day-to-day commands (cheat sheet)

```bash
# Tests
uv run pytest

# Full quality check
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest

# Crawl
uv run civic-slm crawl --jurisdiction san-clemente --max 20

# Eval against any served model
uv run civic-slm eval run --model <id> --bench factuality \
    --bench-file data/eval/civic_factuality.jsonl --base-url http://127.0.0.1:8080

# Train (with smoke first)
uv run civic-slm train cpt --max-iters 100
uv run civic-slm train cpt

# SFT review
uv run python scripts/review_sft.py

# Release
uv run python scripts/merge_quantize.py --adapter-dir artifacts/qwen-civic-sft \
    --base-model mlx-community/Qwen2.5-7B-Instruct-4bit --version v1
```

## What to watch for

- **Eval harness drift.** If you change a scorer, re-run the baseline before training. A scorer change masks model regressions.
- **Synth quality.** Curate the first 500 by hand. If you see patterns (the model always answers in the same shape, refuses too aggressively, hallucinates dates), the prompt template needs tuning. Rotate the `prompt_sha` and re-generate.
- **Memory pressure during training.** Drop batch size first, then increase grad accumulation, before reducing LoRA rank.
- **Ground truth in extraction.** The base model nests under `staff_report` instead of producing flat JSON. Make sure your synth examples enforce the flat schema explicitly, or you'll bake the wrong shape in.

That's the whole loop.
