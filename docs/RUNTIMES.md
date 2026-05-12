# Runtimes

civic-slm standardizes on **LM Studio** as the local-inference runtime. Eval, synth (`CIVIC_SLM_LLM_BACKEND=local`), the side-by-side judge, and the web playground all speak LM Studio's OpenAI-compatible endpoint on `http://127.0.0.1:1234`.

Training is a separate concern (see [Training](#training-still-shells-out-to-mlx-lm) below); the LM Studio assumption applies to inference only.

## TL;DR — set it up once

1. Install LM Studio (https://lmstudio.ai) and open it.
2. Search the model browser, download **`qwen3.6-27b-ud-mlx`** (this is the project's base/candidate model). Optionally also download **`gemma-4-31b-it-mlx`** for the web playground's alternate-model slot and for comparator evals.
3. Developer tab → **Start Server** (defaults to port 1234). Make sure both models are listed as loaded if you plan to run the side-by-side judge.
4. From this repo:

   ```bash
   set -a; source .envrc.lmstudio; set +a
   uv run civic-slm doctor          # green candidate runtime row = ready
   ```

The `.envrc.lmstudio` file in the repo root contains every `CIVIC_SLM_*` variable preset for this setup. Source it once per shell or wire it up with direnv.

## What civic-slm reads from env

| Variable                    | Default                          | What it controls                                                                         |
| --------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------- |
| `CIVIC_SLM_CANDIDATE_URL`   | `http://127.0.0.1:1234`          | Base URL of LM Studio. Override if you've moved it off the default port.                 |
| `CIVIC_SLM_CANDIDATE_MODEL` | `qwen3.6-27b-ud-mlx`             | The model id eval requests reference. Use whatever LM Studio reports in `/v1/models`.    |
| `CIVIC_SLM_TEACHER_URL`     | `http://127.0.0.1:1234`          | Same LM Studio server; only the model id differs.                                        |
| `CIVIC_SLM_TEACHER_MODEL`   | `qwen3.6-27b-ud-mlx`             | The model used for synth (local backend) and the side-by-side judge.                     |
| `CIVIC_SLM_LLM_BACKEND`     | `anthropic`                      | Set to `local` to route synth/judge/crawler through LM Studio instead of Anthropic.      |
| `CIVIC_SLM_STRICT_LOCAL`    | unset                            | Truthy → backends refuse to call Anthropic at runtime, even if the key is loaded.        |
| `CIVIC_SLM_LOCAL_LLM_URL`   | `http://127.0.0.1:1234`          | URL the `LocalBackend` (synth + judge) hits. Mirrors LM Studio's default.                |
| `CIVIC_SLM_LOCAL_LLM_MODEL` | `qwen3.6-27b-ud-mlx`             | Model id passed by the `LocalBackend`.                                                   |
| `CIVIC_SLM_TIMEOUT_S`       | `120`                            | Per-request timeout for the chat client. Bump for long-context evals on slower hardware. |
| `CIVIC_SLM_WHISPER_MODEL`   | `mlx-community/whisper-large-v3` | Used by the optional `crawl-videos` ASR fallback only.                                   |

## Strict-local mode (zero API spend, with proof)

Two switches buy you a verifiable zero-spend run:

```bash
export CIVIC_SLM_LLM_BACKEND=local
export CIVIC_SLM_STRICT_LOCAL=1
```

`select_backend()` (synth + judge) and `agent_llm()` (browser-use crawler) consult `is_strict_local()` at runtime. Misconfigured backend? `RuntimeError` instead of a silent token bill.

The doctor check (`civic-slm doctor --strict-local`) is the one-shot audit; it exits non-zero if anything in the env could still reach Anthropic. The env tripwire is the always-on enforcement.

## Side-by-side comparator

The `side_by_side` eval pits the candidate against a larger comparator with a pairwise judge. Default setup:

- **Candidate:** `qwen3.6-27b-ud-mlx` (the project base).
- **Comparator:** `gemma-4-31b-it-mlx` (or any other model loaded in LM Studio).
- **Judge:** `qwen3.6-27b-ud-mlx` (same as candidate, with A/B position swap to mitigate self-bias) or Anthropic if `CIVIC_SLM_LLM_BACKEND=anthropic`.

Load both candidate and comparator in LM Studio (it serves multiple models on the same port; differ only by model id). Verify with `civic-slm doctor --teacher`.

```bash
export CIVIC_SLM_CANDIDATE_MODEL=qwen3.6-27b-ud-mlx
export CIVIC_SLM_TEACHER_MODEL=gemma-4-31b-it-mlx
civic-slm doctor --teacher
civic-slm eval side-by-side --candidate-model qwen3.6-27b-ud-mlx
```

## Training (still shells out to mlx-lm)

Training is the one place the project doesn't talk to LM Studio. `civic-slm train cpt | sft | dpo` subprocesses out to `mlx_lm.lora` / `mlx_lm.dpo` because LoRA on Apple Silicon currently has no production alternative — Axolotl/Unsloth/TRL are CUDA-first, llama.cpp's `finetune` is CPU-only.

You don't run `mlx_lm.lora` yourself. The trainer wrappers handle it. The base model in `configs/{cpt,sft,dpo}.yaml` is the HF repo path to download for fine-tuning (the same model LM Studio serves at inference time).

If you eventually move training to a Linux/CUDA box, swap in Axolotl — the rest of the pipeline (crawl, process, synth, eval, merge) doesn't care which trainer produced the adapter.

## Quantization

`scripts/merge_quantize.py` fuses the final adapter and emits both **MLX 4-bit** (Apple Silicon native) and **GGUF Q5_K_M** (for whoever wants to run civic-slm on a non-Mac runtime). It calls `mlx_lm.fuse` + `mlx_lm.convert` for the MLX artifact and llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` for the GGUF artifact. As with training, you don't run these directly — the script does.

For the GGUF path, `brew install llama.cpp` once and you're set.
