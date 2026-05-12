"""Runtime defaults for civic-slm local inference.

The project standardizes on **LM Studio** for local model serving. Every
inference call — eval, synth (when `CIVIC_SLM_LLM_BACKEND=local`), the
side-by-side judge, and the web playground — speaks LM Studio's
OpenAI-compatible `/v1/chat/completions` endpoint on port 1234.

Defaults point at `http://127.0.0.1:1234` with model id `qwen3.6-27b-ud-mlx`.
Override per shell when you've loaded something else:

    export CIVIC_SLM_CANDIDATE_URL=http://127.0.0.1:1234
    export CIVIC_SLM_CANDIDATE_MODEL=qwen3.6-27b-ud-mlx
    export CIVIC_SLM_TEACHER_MODEL=gemma-4-31b-it-mlx

See `.envrc.lmstudio` for a sourceable env block.
"""

from __future__ import annotations

import os

# --- env-driven defaults ------------------------------------------------------


def candidate_url() -> str:
    """Where to reach the candidate model.

    Default `http://127.0.0.1:1234` matches LM Studio's developer-server port.
    """
    return os.environ.get("CIVIC_SLM_CANDIDATE_URL", "http://127.0.0.1:1234")


def candidate_model() -> str:
    """The `model` field sent in chat-completion requests for the candidate.

    Default `qwen3.6-27b-ud-mlx` matches what LM Studio publishes by that
    name. Override per shell if you've loaded a different model.
    """
    return os.environ.get("CIVIC_SLM_CANDIDATE_MODEL", "qwen3.6-27b-ud-mlx")


def teacher_url() -> str:
    """Where to reach the teacher model (synth + judge).

    LM Studio multi-hosts on a single port; the teacher differs from the
    candidate only by `CIVIC_SLM_TEACHER_MODEL`.
    """
    return os.environ.get("CIVIC_SLM_TEACHER_URL", "http://127.0.0.1:1234")


def teacher_model() -> str:
    return os.environ.get("CIVIC_SLM_TEACHER_MODEL", "qwen3.6-27b-ud-mlx")


# --- strict-local tripwire ---------------------------------------------------

_STRICT_LOCAL_TRUTHY = {"1", "true", "yes", "on"}


def is_strict_local() -> bool:
    """Return True iff `CIVIC_SLM_STRICT_LOCAL` is set to a truthy value.

    When strict-local is on, every code path that could otherwise call
    Anthropic raises a `RuntimeError` instead. The check is consulted by
    `civic_slm.llm.backend.select_backend` (synth + judge) and by
    `civic_slm.ingest.recipes._browser.agent_llm` (browser-use crawler),
    so a misconfigured `CIVIC_SLM_LLM_BACKEND` can't silently spend tokens.
    """
    return os.environ.get("CIVIC_SLM_STRICT_LOCAL", "").strip().lower() in _STRICT_LOCAL_TRUTHY
