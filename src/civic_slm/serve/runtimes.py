"""Runtime-agnostic serving helpers.

The pipeline only assumes one thing about how you serve models: an
**OpenAI-compatible `/v1/chat/completions` endpoint**. Every popular Mac
runtime exposes one. Pick whichever you like — we don't lock you in.

Pre-baked presets cover the common defaults so you don't have to memorize
ports and model name conventions:

  | Runtime    | Default URL              | Model id convention                   |
  | ---------- | ------------------------ | ------------------------------------- |
  | mlx        | http://127.0.0.1:8080    | hf id, e.g. `mlx-community/...`       |
  | llama_cpp  | http://127.0.0.1:8080    | server reports `default` unless --alias |
  | ollama     | http://127.0.0.1:11434   | tag id, e.g. `qwen2.5:7b`             |
  | lm_studio  | http://127.0.0.1:1234    | gguf basename or HF id                |
  | openai     | $OPENAI_BASE_URL or HTTP | whatever the server expects           |

If you stand up something else, set `CIVIC_SLM_CANDIDATE_URL` /
`CIVIC_SLM_CANDIDATE_MODEL` (and `_TEACHER_URL` / `_TEACHER_MODEL` for the
synth+judge backend) to point at it. No code changes needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class Runtime(StrEnum):
    MLX = "mlx"
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class RuntimePreset:
    runtime: Runtime
    base_url: str
    model_hint: str  # human-readable example, not the actual model id
    start_command: str  # one-line shell example


_PRESETS: dict[Runtime, RuntimePreset] = {
    Runtime.MLX: RuntimePreset(
        runtime=Runtime.MLX,
        base_url="http://127.0.0.1:8080",
        model_hint="mlx-community/Qwen2.5-7B-Instruct-4bit",
        start_command=(
            "uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080"
        ),
    ),
    Runtime.LLAMA_CPP: RuntimePreset(
        runtime=Runtime.LLAMA_CPP,
        base_url="http://127.0.0.1:8080",
        model_hint="(any GGUF; server reports `default` by default)",
        start_command="llama-server -m ~/models/qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 --port 8080",  # noqa: E501
    ),
    Runtime.OLLAMA: RuntimePreset(
        runtime=Runtime.OLLAMA,
        base_url="http://127.0.0.1:11434",
        model_hint="qwen2.5:7b-instruct",
        start_command="ollama serve  # then: ollama pull qwen2.5:7b-instruct",
    ),
    Runtime.LM_STUDIO: RuntimePreset(
        runtime=Runtime.LM_STUDIO,
        base_url="http://127.0.0.1:1234",
        model_hint="qwen2.5-7b-instruct (whatever you loaded in the GUI)",
        start_command="# In LM Studio: Developer → Start Server (defaults to port 1234)",
    ),
    Runtime.OPENAI_COMPAT: RuntimePreset(
        runtime=Runtime.OPENAI_COMPAT,
        base_url="http://127.0.0.1:8080",
        model_hint="(set CIVIC_SLM_CANDIDATE_MODEL to whatever your server expects)",
        start_command="# Use whatever runtime you like — set CIVIC_SLM_CANDIDATE_URL to its base URL",  # noqa: E501
    ),
}


def preset(runtime: Runtime) -> RuntimePreset:
    return _PRESETS[runtime]


# --- env-driven defaults ------------------------------------------------------


def candidate_url() -> str:
    """Where to reach the candidate model (the 7B base or fine-tune you're scoring).

    Default `http://127.0.0.1:8080` matches MLX-LM and llama.cpp out of the box.
    """
    return os.environ.get("CIVIC_SLM_CANDIDATE_URL", "http://127.0.0.1:8080")


def candidate_model() -> str:
    """The `model` field sent in chat-completion requests for the candidate."""
    return os.environ.get(
        "CIVIC_SLM_CANDIDATE_MODEL",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    )


def teacher_url() -> str:
    """Where to reach the teacher model (synth + judge). Defaults to llama.cpp port."""
    return os.environ.get("CIVIC_SLM_TEACHER_URL", "http://127.0.0.1:8081")


def teacher_model() -> str:
    return os.environ.get("CIVIC_SLM_TEACHER_MODEL", "default")
