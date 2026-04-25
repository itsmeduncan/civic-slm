# Bring your own runtime

The pipeline only assumes one thing about how you serve models: an **OpenAI-compatible `/v1/chat/completions` endpoint**. That's it. Use whichever runtime fits your taste — MLX-LM, llama.cpp, Ollama, LM Studio, vLLM (Linux), or anything else that speaks the OpenAI shape.

This doc has copy-paste setup for each, plus a streamlined model matrix so you know exactly which weights to download.

## TL;DR — minimum viable setup

You only need **one model** to start: **Qwen2.5-7B-Instruct (4-bit)**. Pick a runtime, point civic-slm at it, run the baseline.

```bash
# Option A — MLX-LM (most native on Apple Silicon)
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# Option B — Ollama (most ergonomic; great if you already use it)
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve   # exposes 11434

# Option C — LM Studio (GUI; nice for non-CLI users)
# In the app: search "Qwen2.5 7B Instruct", download the 4-bit GGUF,
# Developer tab → Start Server (defaults to port 1234)

# Option D — llama.cpp directly
brew install llama.cpp
llama-server -m ~/models/qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 --port 8080
```

Then tell civic-slm where to find it:

```bash
# MLX or llama.cpp on default port — no env needed (8080 is the default)
# Ollama:
export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:11434
export CIVIC_SLM_CANDIDATE_MODEL=qwen2.5:7b-instruct-q4_K_M
# LM Studio:
export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:1234
export CIVIC_SLM_CANDIDATE_MODEL=qwen2.5-7b-instruct  # whatever you loaded
```

Verify with the doctor command:

```bash
civic-slm doctor
```

If you see a green row for "candidate runtime", you're done. Run the baseline (Step 1 in [USAGE.md](USAGE.md)).

## The streamlined model matrix

| Stage                                       | Required?                             | Model                                                           | Disk                      | Why                                                                                                                                                                                        |
| ------------------------------------------- | ------------------------------------- | --------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Candidate** (eval target, fine-tune base) | always                                | Qwen2.5-7B-Instruct, 4-bit                                      | ~4.5GB                    | The thing we're fine-tuning.                                                                                                                                                               |
| **Teacher** (synth + judge)                 | only if `CIVIC_SLM_LLM_BACKEND=local` | Qwen2.5-72B-Instruct, Q4_K_M GGUF (or 32B if you have less RAM) | ~40GB (72B) / ~20GB (32B) | Generates synthetic SFT data and judges side-by-side. If you're OK paying the Anthropic API, you don't need this — set `CIVIC_SLM_LLM_BACKEND=anthropic` (the default) and Claude does it. |
| **BGE reranker** (factuality scoring)       | not yet                               | `BAAI/bge-reranker-large`                                       | ~2GB                      | Planned upgrade for the factuality scorer. Current word-overlap proxy is fine for v0.                                                                                                      |

So the realistic minimums:

- **Cheapest setup (any-Mac):** 1 model — Qwen2.5-7B (~4.5GB) + Anthropic API for synth/judge. ~$10–20 in API credits gets you to v0.
- **Fully-local (≥64GB unified RAM):** 2 models — Qwen2.5-7B + Qwen2.5-32B-Instruct (~25GB total). $0 API.
- **Best-quality fully-local (≥96GB unified RAM):** 2 models — Qwen2.5-7B + Qwen2.5-72B-Instruct (~45GB total). Closest to Claude-quality teacher.

Everything else from the original CLAUDE.md spec — vLLM, AWQ, Axolotl — is optional and not required at v0.

## Per-runtime setup

### MLX-LM (Apple Silicon native)

Best-in-class generation speed on M-series. Required anyway for **training and quantization** (the only training stack that runs on Apple Silicon GPUs). For inference, it's a fine choice but not the only one.

```bash
uv sync --extra train     # installs mlx, mlx-lm
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

civic-slm defaults match this: `CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:8080`, `CIVIC_SLM_CANDIDATE_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit`.

### Ollama (most ergonomic)

If you already run Ollama, you're done in two commands. It auto-pulls weights and exposes the OpenAI shape on port 11434.

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve   # if not already running
```

```bash
export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:11434
export CIVIC_SLM_CANDIDATE_MODEL=qwen2.5:7b-instruct-q4_K_M
civic-slm doctor
```

For the teacher slot:

```bash
ollama pull qwen2.5:72b-instruct-q4_K_M
export CIVIC_SLM_TEACHER_URL=http://127.0.0.1:11434
export CIVIC_SLM_TEACHER_MODEL=qwen2.5:72b-instruct-q4_K_M
export CIVIC_SLM_LLM_BACKEND=local
```

(Yes, Ollama can serve both candidate and teacher on the same port — it routes by model id.)

### LM Studio (GUI)

Nice if you prefer not to fight the terminal for serving.

1. Install LM Studio.
2. Search "Qwen2.5 7B Instruct" in the model browser, download the **4-bit GGUF**.
3. **Developer** tab → **Start Server** → defaults to port 1234.
4. Note the model name shown at the top of the server panel (e.g. `qwen2.5-7b-instruct`).

```bash
export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:1234
export CIVIC_SLM_CANDIDATE_MODEL=qwen2.5-7b-instruct
civic-slm doctor
```

### llama.cpp directly

Lightest dependency. Required for the GGUF quantization path (`scripts/merge_quantize.py`) — install once and you're set for both serving and quantization.

```bash
brew install llama.cpp
# download a GGUF, e.g.:
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir ~/models
llama-server -m ~/models/qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 --port 8080
```

Defaults match civic-slm's defaults — no env needed.

### vLLM (Linux/CUDA only)

If you're running this on a CUDA box, vLLM is the fastest option. We don't ship vLLM helpers since v0 is Apple-Silicon-first, but the integration is identical: `vllm serve <model>` exposes `/v1/chat/completions`, point `CIVIC_SLM_CANDIDATE_URL` at it.

### Anything else (OpenAI, Together, Fireworks, custom)

If your runtime exposes `/v1/chat/completions` and accepts a Bearer token, civic-slm can use it. Set:

```bash
export CIVIC_SLM_CANDIDATE_URL=https://api.example.com
export CIVIC_SLM_CANDIDATE_MODEL=whatever-the-server-expects
# If it requires auth, the api_key arg in serve/client.py defaults to "not-needed";
# patch that to read from env if you need real auth.
```

## What civic-slm reads from env

| Var                         | Default                                     | Used by                                                                                             |
| --------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `CIVIC_SLM_CANDIDATE_URL`   | `http://127.0.0.1:8080`                     | `civic-slm eval run`, `civic-slm eval side-by-side`, `civic-slm doctor`                             |
| `CIVIC_SLM_CANDIDATE_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit`    | same                                                                                                |
| `CIVIC_SLM_TEACHER_URL`     | `http://127.0.0.1:8081`                     | `civic-slm eval side-by-side` (comparator), local-backend synth + judge                             |
| `CIVIC_SLM_TEACHER_MODEL`   | `default`                                   | same                                                                                                |
| `CIVIC_SLM_LLM_BACKEND`     | `anthropic`                                 | synth + judge + crawler: `local` to use the teacher URL, `anthropic` to use the SDK                 |
| `CIVIC_SLM_LOCAL_LLM_URL`   | `$CIVIC_SLM_TEACHER_URL` if local backend   | `synth/generate.py`, `eval/judge.py`                                                                |
| `CIVIC_SLM_LOCAL_LLM_MODEL` | `$CIVIC_SLM_TEACHER_MODEL` if local backend | same                                                                                                |
| `CIVIC_SLM_STRICT_LOCAL`    | unset                                       | runtime tripwire — any of `1\|true\|yes\|on` makes synth/judge/crawler refuse Anthropic (see below) |
| `CIVIC_SLM_TIMEOUT_S`       | `120` (ChatClient), `600` (Backend)         | HTTP timeouts for the chat client and synth/judge backends                                          |
| `CIVIC_SLM_WHISPER_MODEL`   | `mlx-community/whisper-large-v3-turbo`      | ASR fallback during `civic-slm crawl-videos`                                                        |

Set what you need; everything has a sensible default. `civic-slm doctor` prints what it's actually going to use.

## Strict-local mode (zero API spend, with proof)

If you want to guarantee the pipeline can't spend a paid token — useful before a multi-hour synth job, or when running on a fresh machine where you're not sure what's in env — flip the strict-local tripwire:

```bash
export CIVIC_SLM_STRICT_LOCAL=1
export CIVIC_SLM_LLM_BACKEND=local
```

Two things happen:

1. **Runtime guard.** Every code path that could otherwise call Anthropic (`synth.generate`, `eval.judge`, the browser-use crawler) calls `select_backend()` or `agent_llm()`, both of which now raise `RuntimeError` if the backend would resolve to anything other than `local`. The error names the env var to fix. No silent fallthrough.
2. **Doctor audit.** `civic-slm doctor --strict-local` runs the full env audit and _fails_ (exit code 1) if anything could reach Anthropic. Specifically:
   - `CIVIC_SLM_LLM_BACKEND` must equal `local`. (Default `anthropic` fails.)
   - `ANTHROPIC_API_KEY` must not be loaded by the config. Even if `BACKEND=local` overrides it, leaving the key in `~/.config/civic-slm/.env` is a footgun and gets flagged.
   - Teacher URL must respond on `/v1/chat/completions`. (Non-strict: warning. Strict: failure.)
   - Candidate and teacher URLs should look local (`127.0.0.1`, `localhost`, `*.local`, RFC 1918 ranges). Public URLs warn but don't fail — Tailscale, ZeroTier, and private DNS names look non-local but are legitimate.

A typical zero-spend session:

```bash
# 1. Stand up the teacher (one of):
llama-server -m ~/models/qwen2.5-72b-instruct-q4_k_m.gguf -c 8192 --port 8081
# or: ollama serve  (with the right model pulled)

# 2. Stand up the candidate (one of):
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# 3. Lock the env.
unset ANTHROPIC_API_KEY
export CIVIC_SLM_STRICT_LOCAL=1
export CIVIC_SLM_LLM_BACKEND=local

# 4. Verify.
uv run civic-slm doctor --strict-local
# Expect: every row OK or WARN (never FAIL); exit 0.

# 5. Run anything. Synth, side-by-side, crawl — none of it can reach Anthropic.
```

If anything's miswired, you get a single actionable error before you've spent a token.

### What strict-local does NOT do

- It doesn't prevent `mlx-whisper` from downloading model weights from Hugging Face on first ASR call. That's a one-time download, not API spend, and it only triggers in the `crawl-videos` ASR fallback path (which only fires when YouTube has no captions).
- It doesn't stop you from setting `CIVIC_SLM_LLM_BACKEND=anthropic` _and_ `STRICT_LOCAL=1` simultaneously — the next call into synth/judge/crawler will simply raise. That's the intended behavior.

## Standing up the 72B comparator (for `side_by_side`)

The `side_by_side` benchmark scores the candidate against a comparator
model — by design, that comparator is `Qwen2.5-72B-Instruct` so we can
make the "approaches 72B on civic tasks" claim with evidence. Running a
72B at home is expensive but tractable on a Mac with ≥64GB unified
memory; the recommended path is llama.cpp's `llama-server` serving a Q4
GGUF on a different port from the candidate.

```bash
# Download once (~40GB). Pick whichever Qwen2.5-72B GGUF mirror you trust.
huggingface-cli download \
    bartowski/Qwen2.5-72B-Instruct-GGUF Qwen2.5-72B-Instruct-Q4_K_M.gguf \
    --local-dir ~/models

# Stand it up on port 8081 (the candidate is on 8080).
llama-server \
    -m ~/models/Qwen2.5-72B-Instruct-Q4_K_M.gguf \
    -c 8192 \
    --port 8081 \
    --n-gpu-layers -1
```

Point civic-slm at it:

```bash
export CIVIC_SLM_TEACHER_URL=http://127.0.0.1:8081
export CIVIC_SLM_TEACHER_MODEL=default
```

Verify before launching the bench:

```bash
uv run civic-slm doctor --teacher
```

Then run the side-by-side eval:

```bash
uv run civic-slm eval side-by-side --candidate-model base-qwen2.5-7b
```

If the comparator is unreachable, the runner exits cleanly with a
`ComparatorMissingError` pointing back at this section, rather than
crashing on the first chat call after an entire candidate-side warmup.

### Hardware reality check

- **Disk:** ~40GB for the Q4 GGUF. Q5_K_M is ~50GB and noticeably
  slower; Q4 is the right tradeoff for a comparator.
- **RAM:** ≥48GB unified is the floor (you'll swap); ≥64GB is
  comfortable. M-series Pro/Max with 64-128GB is the sweet spot.
- **Throughput:** 5-15 tok/s on M2 Pro 64GB, 15-25 tok/s on M2 Max
  96GB. A 100-example side_by_side run takes 1-3 hours; budget
  accordingly.
- **Power:** the bench is a sustained load. Plug in.

If 72B is out of reach, you can use Qwen2.5-32B as the comparator and
note it in your model card; the headline claim shifts from "approaches
72B" to "approaches 32B," which is still meaningful.

## Why we still recommend MLX for training

Inference is interchangeable across runtimes. Training is not. On Apple Silicon, **MLX-LM is the only mature path** for LoRA + DPO. The alternatives:

- Axolotl, Unsloth, TRL — all CUDA-first, no Metal.
- llama.cpp `finetune` — CPU only, slow, limited.
- mlx-lm — first-party Apple, tight integration with the Metal compiler, supports LoRA, DPO, and quantization out of the box.

So civic-slm's `train cpt | sft | dpo` shells out to `mlx_lm.lora` / `mlx_lm.dpo`, which is the right call on a Mac. If you move to a Linux/CUDA box later, swap in Axolotl — the rest of the pipeline doesn't care.

## Why we still recommend llama.cpp for quantization

The release artifact is GGUF Q5_K_M (so Ollama / LM Studio / llama.cpp users can run it) plus MLX-q4 (so MLX users can). Producing both currently requires both `mlx_lm.fuse` + `mlx_lm.convert` and llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`. `scripts/merge_quantize.py` does both for you.
