"""Single source of truth for what `--model <foo>` means.

Why this file exists: before it, the project had two parallel ideas of "model":
the project-side artifact-directory label (e.g. `base-qwen3.6-27b`, used in
`artifacts/evals/<label>/`) and the runtime-side served-model name that LM
Studio publishes via `/v1/models` (e.g. `qwen3.6-27b-ud-mlx`). They were
plumbed through five overlapping env vars and three CLI flags, and they could
silently disagree — `--model base-qwen3.6-27b` would happily run against
`gemma-4-31b-it-mlx` if `$CIVIC_SLM_CANDIDATE_MODEL` pointed there. Baselines
got mislabeled, comparator and candidate could end up identical, and "I asked
for X, did I get X?" became a real question.

The registry below collapses both ideas into one named entry. `--model
base-qwen3.6-27b` resolves through `MODELS["base-qwen3.6-27b"]` to *both* the
artifact-dir label *and* the served-model name in a single lookup. There is
no separate `--served-model` flag any more, and no separate env var either.

Adding a model: add one entry. Renaming an LM Studio model: change one
string, every call site keeps working.

Unregistered labels are accepted as a fallback (`Model(label=x, served_name=x)`)
so one-off testing of an arbitrary LM Studio name still works without
ceremony — but the label and served name are guaranteed identical, so silent
divergence is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """One entry in the project-side model registry.

    `label` is what callers say (`--model base-qwen3.6-27b`) and where artifacts
    land (`artifacts/evals/base-qwen3.6-27b/`). `served_name` is what we send in
    the `model` field of the chat-completion HTTP body — it must match what the
    inference server publishes in `/v1/models`.
    """

    label: str
    served_name: str


# Stable project-side labels → LM Studio served names.
#
# The label is the canonical identity in code, docs, MODEL_CARD, and artifact
# paths. The served_name only matters at the moment we make an HTTP call.
# Adding a new candidate / comparator / fine-tune: add one row.
MODELS: dict[str, Model] = {
    # The project base — Qwen 3.6 27B Instruct (MLX 4-bit) per LM Studio's catalog.
    "base-qwen3.6-27b": Model(
        label="base-qwen3.6-27b",
        served_name="qwen3.6-27b-ud-mlx",
    ),
    # The previous base, kept for backwards comparability of older evals.
    "base-qwen2.5-7b": Model(
        label="base-qwen2.5-7b",
        served_name="qwen2.5-7b-instruct-mlx",
    ),
    # Default side-by-side comparator — Gemma 4 31B Instruct (MLX) per LM Studio.
    "comparator-gemma-4-31b": Model(
        label="comparator-gemma-4-31b",
        served_name="gemma-4-31b-it-mlx",
    ),
    # Standalone Gemma label — same served binary, used when Gemma itself is
    # the candidate (e.g., to publish a Gemma baseline).
    "base-gemma-4-31b": Model(
        label="base-gemma-4-31b",
        served_name="gemma-4-31b-it-mlx",
    ),
    # Placeholder until v1 ships; bumped to point at the real fine-tune at merge time.
    # v1 fused artifact, served by `mlx_lm.server --model artifacts/civic-slm-v1-fused`.
    # mlx_lm.server identifies the model by its local path; the served_name MUST
    # match the path passed to `--model` (no canonical HF id yet).
    "civic-slm-v1": Model(
        label="civic-slm-v1",
        served_name="artifacts/civic-slm-v1-fused",
    ),
    # v1.1 multi-jurisdiction retrain. 7-juris CPT corpus (495 chunks) +
    # 5-juris SFT corpus (2702/300 train/valid examples from sc/seattle/boston/
    # denver/cook-county). Trained 2026-05-14 → 2026-05-15.
    # mlx_lm.server identifies the model by the path it was loaded under, so
    # launch the server from the project root with `--model
    # artifacts/multi-v11-mlx-q4` so this label matches across machines.
    "civic-slm-v11": Model(
        label="civic-slm-v11",
        served_name="artifacts/multi-v11-mlx-q4",
    ),
}


class ModelLookupError(KeyError):
    """Raised when a label is requested with `strict=True` but isn't registered."""


def resolve(label: str, *, strict: bool = False) -> Model:
    """Map a project-side label to its `Model` entry.

    Default behavior: registered labels go through the table; unregistered
    labels fall back to `Model(label=label, served_name=label)` so one-off
    testing of an arbitrary LM Studio name still works. The label and served
    name are identical in the fallback case — there is no way for them to
    silently disagree.

    Pass `strict=True` to refuse the fallback and raise `ModelLookupError`
    instead. Useful for code paths that should only accept curated labels.
    """
    if label in MODELS:
        return MODELS[label]
    if strict:
        known = ", ".join(sorted(MODELS))
        raise ModelLookupError(
            f"model label {label!r} is not registered. Known labels: {known}. "
            "Add it to civic_slm.serve.models.MODELS or pass strict=False."
        )
    return Model(label=label, served_name=label)


def known_labels() -> list[str]:
    """Sorted list of registered labels — for `--help` text and doctor output."""
    return sorted(MODELS)
