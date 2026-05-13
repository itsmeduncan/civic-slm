# Runtimes

civic-slm standardizes on **LM Studio** as the local-inference runtime. Eval, synth (`CIVIC_SLM_LLM_BACKEND=local`), the side-by-side judge, and the web playground all speak LM Studio's OpenAI-compatible endpoint on `http://127.0.0.1:1234`.

Training is a separate concern (see [Training](#training-still-shells-out-to-mlx-lm) below); the LM Studio assumption applies to inference only.

## TL;DR — set it up once

1. Install LM Studio (https://lmstudio.ai) and open it.
2. Search the model browser, download **`qwen3.6-27b-ud-mlx`** (the project's base model). Optionally also download **`gemma-4-31b-it-mlx`** for the side-by-side comparator and the web playground's alternate slot.
3. Developer tab → **Start Server** (defaults to port 1234). Make sure both models are listed as loaded if you plan to run side-by-side evals.
4. From this repo:

   ```bash
   set -a; source .envrc.lmstudio; set +a
   uv run civic-slm doctor          # green candidate row = ready
   ```

The `.envrc.lmstudio` file in the repo root presets the two env vars below. Source it once per shell or wire it up with direnv.

## Configuration model

There are exactly **two env vars** for runtime selection, plus the strict-local tripwire:

| Variable                  | Default                          | What it controls                                                                                       |
| ------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `CIVIC_SLM_LM_STUDIO_URL` | `http://127.0.0.1:1234`          | Where the OpenAI-compatible inference server is listening. One URL serves every role.                  |
| `CIVIC_SLM_DEFAULT_MODEL` | `base-qwen3.6-27b`               | Project-side **label** (not a served name). Resolves through the registry to a served name. See below. |
| `CIVIC_SLM_LLM_BACKEND`   | `anthropic`                      | Set to `local` to route synth/judge/crawler through LM Studio instead of Anthropic.                    |
| `CIVIC_SLM_STRICT_LOCAL`  | unset                            | Truthy → backends refuse to call Anthropic at runtime, even if the key is loaded.                      |
| `CIVIC_SLM_TIMEOUT_S`     | `600`                            | Per-request timeout for the chat client. Bump for long-context evals on slower hardware.               |
| `CIVIC_SLM_WHISPER_MODEL` | `mlx-community/whisper-large-v3` | Used by the optional `crawl-videos` ASR fallback only.                                                 |

Anything else that looks like it configures a model is **gone**. The previous setup (`CIVIC_SLM_CANDIDATE_URL` / `_MODEL`, `CIVIC_SLM_TEACHER_URL` / `_MODEL`, `CIVIC_SLM_LOCAL_LLM_URL` / `_MODEL`, `CIVIC_SLM_GEMMA_MODEL`, `CIVIC_SLM_CIVIC_MODEL`) let `--model X` silently run against a different served model. CLI entry points now hard-error if any of those names are still in your environment, so update `.envrc` if you see the failure.

## The model registry

`src/civic_slm/serve/models.py` is the single source of truth for the project's model labels. Each entry maps a stable label (e.g. `base-qwen3.6-27b`) to the served-model name LM Studio publishes (e.g. `qwen3.6-27b-ud-mlx`). When you pass `--model base-qwen3.6-27b`, both the artifact directory and the served name come from one lookup — they cannot disagree.

Adding a new model: append one entry to `MODELS`. Renaming an LM Studio model: change one string. Web playground keeps a TS mirror at `web/src/lib/models.ts` (separate process, must be kept in sync — see the comment at the top of that file).

Default labels:

| Label                    | Served name               | Role                                   |
| ------------------------ | ------------------------- | -------------------------------------- |
| `base-qwen3.6-27b`       | `qwen3.6-27b-ud-mlx`      | Project base / default candidate.      |
| `base-qwen2.5-7b`        | `qwen2.5-7b-instruct-mlx` | Previous base, kept for comparability. |
| `comparator-gemma-4-31b` | `gemma-4-31b-it-mlx`      | Default side-by-side comparator.       |
| `base-gemma-4-31b`       | `gemma-4-31b-it-mlx`      | Same binary; for Gemma-as-candidate.   |
| `civic-slm-v1`           | (placeholder)             | Will be repointed when v1 ships.       |

Unregistered labels still work — they resolve to `Model(label=x, served_name=x)`, so a one-off `--model some-experimental-mlx-build` still works without ceremony, but with no way for the label and served name to diverge.

## Strict-local mode (zero API spend, with proof)

```bash
export CIVIC_SLM_LLM_BACKEND=local
export CIVIC_SLM_STRICT_LOCAL=1
```

`select_backend()` (synth + judge) and `agent_llm()` (browser-use crawler) consult `is_strict_local()` at runtime. Misconfigured backend? `RuntimeError` instead of a silent token bill.

The doctor check (`civic-slm doctor --strict-local`) is the one-shot audit; it exits non-zero if anything in the env could still reach Anthropic. The env tripwire is the always-on enforcement.

## Side-by-side comparator

The `side-by-side` eval pits the candidate against a comparator with a pairwise judge. Default setup:

- **Candidate:** label `base-qwen3.6-27b` → served `qwen3.6-27b-ud-mlx`.
- **Comparator:** label `comparator-gemma-4-31b` → served `gemma-4-31b-it-mlx`.
- **Judge:** Claude Sonnet 4.6 by default (needs `ANTHROPIC_API_KEY`).

Load both candidate and comparator in LM Studio (it serves multiple models on the same port — differ only by model id). Verify both are reachable with `civic-slm doctor --comparator comparator-gemma-4-31b`.

```bash
civic-slm doctor --candidate base-qwen3.6-27b --comparator comparator-gemma-4-31b
civic-slm eval side-by-side \
  --candidate base-qwen3.6-27b \
  --comparator comparator-gemma-4-31b
```

## Training (still shells out to mlx-lm)

Training is the one place the project doesn't talk to LM Studio. `civic-slm train cpt | sft | dpo` subprocesses out to `mlx_lm.lora` / `mlx_lm.dpo` because LoRA on Apple Silicon currently has no production alternative — Axolotl/Unsloth/TRL are CUDA-first, llama.cpp's `finetune` is CPU-only.

You don't run `mlx_lm.lora` yourself. The trainer wrappers handle it. The base model in `configs/{cpt,sft,dpo}.yaml` is the HF repo path to download for fine-tuning (the same model LM Studio serves at inference time).

If you eventually move training to a Linux/CUDA box, swap in Axolotl — the rest of the pipeline (crawl, process, synth, eval, merge) doesn't care which trainer produced the adapter.

## Quantization

`scripts/merge_quantize.py` fuses the final adapter and emits both **MLX 4-bit** (Apple Silicon native) and **GGUF Q5_K_M** (for whoever wants to run civic-slm on a non-Mac runtime). It calls `mlx_lm.fuse` + `mlx_lm.convert` for the MLX artifact and llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` for the GGUF artifact. As with training, you don't run these directly — the script does.

For the GGUF path, `brew install llama.cpp` once and you're set.
