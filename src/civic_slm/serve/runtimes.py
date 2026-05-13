"""Runtime defaults for civic-slm local inference.

The project standardizes on **LM Studio** for local model serving. Every
inference call — eval, synth (when `CIVIC_SLM_LLM_BACKEND=local`), the
side-by-side judge, and the web playground — speaks LM Studio's
OpenAI-compatible `/v1/chat/completions` endpoint on port 1234.

Two env vars, one job each:

    CIVIC_SLM_LM_STUDIO_URL    # default http://127.0.0.1:1234
    CIVIC_SLM_DEFAULT_MODEL    # default base-qwen3.6-27b (a registry label)

`CIVIC_SLM_DEFAULT_MODEL` is a project-side label, not a served name. It
resolves through `civic_slm.serve.models.resolve()` to the actual served
model. See that module for why this matters and how to add a new model.

The previous five-env-var setup (`CIVIC_SLM_CANDIDATE_URL` / `_MODEL`,
`CIVIC_SLM_TEACHER_URL` / `_MODEL`, `CIVIC_SLM_LOCAL_LLM_URL` / `_MODEL`,
`CIVIC_SLM_GEMMA_MODEL`, `CIVIC_SLM_CIVIC_MODEL`) is gone. If any of those
are still in your environment, `assert_no_deprecated_env()` raises so you
notice immediately instead of silently inheriting the old behavior.
"""

from __future__ import annotations

import os

DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL_LABEL = "base-qwen3.6-27b"


def lm_studio_url() -> str:
    """Where the local OpenAI-compatible inference server is listening.

    Default `http://127.0.0.1:1234` matches LM Studio's developer-server
    port. Override via `CIVIC_SLM_LM_STUDIO_URL` if you've moved it.
    """
    return os.environ.get("CIVIC_SLM_LM_STUDIO_URL", DEFAULT_LM_STUDIO_URL)


def default_model_label() -> str:
    """The default `--model` label when no flag is given.

    Returns a registry label (see `civic_slm.serve.models`), not a served
    name. Default `base-qwen3.6-27b` resolves to LM Studio's
    `qwen3.6-27b-ud-mlx`.
    """
    return os.environ.get("CIVIC_SLM_DEFAULT_MODEL", DEFAULT_MODEL_LABEL)


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


# --- deprecated-env tripwire -------------------------------------------------

_DEPRECATED_ENV: dict[str, str] = {
    "CIVIC_SLM_CANDIDATE_URL": "Use CIVIC_SLM_LM_STUDIO_URL.",
    "CIVIC_SLM_TEACHER_URL": "Use CIVIC_SLM_LM_STUDIO_URL (one URL serves both roles).",
    "CIVIC_SLM_LOCAL_LLM_URL": "Use CIVIC_SLM_LM_STUDIO_URL.",
    "CIVIC_SLM_CANDIDATE_MODEL": (
        "Use --model <label> (or CIVIC_SLM_DEFAULT_MODEL <label>); "
        "see civic_slm.serve.models for the registry."
    ),
    "CIVIC_SLM_TEACHER_MODEL": (
        "Use --comparator <label> on side-by-side, or pass model labels explicitly."
    ),
    "CIVIC_SLM_LOCAL_LLM_MODEL": (
        "Pass an explicit label/served-name into LocalBackend(model=...) at the call site."
    ),
    "CIVIC_SLM_GEMMA_MODEL": (
        "The web playground now reads its slot mapping from web/src/lib/models.ts; "
        "see civic_slm.serve.models for the canonical registry."
    ),
    "CIVIC_SLM_CIVIC_MODEL": (
        "The web playground now reads its slot mapping from web/src/lib/models.ts; "
        "see civic_slm.serve.models for the canonical registry."
    ),
}


def assert_no_deprecated_env() -> None:
    """Raise if any pre-refactor model/URL env vars are still set.

    Called eagerly from CLI entry points so a stale `.envrc` can't silently
    re-introduce the bug this refactor fixed: `--model X` running against a
    different served model because `$CIVIC_SLM_*_MODEL` overrode it.
    """
    found = [(k, _DEPRECATED_ENV[k]) for k in _DEPRECATED_ENV if k in os.environ]
    if not found:
        return
    lines = [f"  {k}: {hint}" for k, hint in found]
    raise RuntimeError(
        "Deprecated environment variables detected:\n"
        + "\n".join(lines)
        + "\n\nThese were removed because they let `--model X` silently run against a "
        "different served model. Update your shell / direnv (.envrc.lmstudio in this "
        "repo is the new template) and re-run."
    )
