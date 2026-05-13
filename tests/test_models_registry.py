"""Unit tests for the model registry and the deprecated-env tripwire.

The whole point of the registry is that `--model X` and the served-name we
send LM Studio cannot disagree. These tests pin that contract.
"""

from __future__ import annotations

import pytest

from civic_slm.serve import models, runtimes


def test_registered_label_resolves_consistently() -> None:
    m = models.resolve("base-qwen3.6-27b")
    assert m.label == "base-qwen3.6-27b"
    assert m.served_name == "qwen3.6-27b-ud-mlx"


def test_unregistered_label_falls_back_to_identity() -> None:
    # Same string on both sides — by design, no silent divergence.
    m = models.resolve("some-experimental-build")
    assert m.label == "some-experimental-build"
    assert m.served_name == "some-experimental-build"


def test_strict_unregistered_raises() -> None:
    with pytest.raises(models.ModelLookupError):
        models.resolve("some-experimental-build", strict=True)


def test_known_labels_includes_default() -> None:
    assert "base-qwen3.6-27b" in models.known_labels()


def test_default_model_label_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIVIC_SLM_DEFAULT_MODEL", raising=False)
    assert runtimes.default_model_label() == "base-qwen3.6-27b"


def test_default_model_label_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_DEFAULT_MODEL", "comparator-gemma-4-31b")
    assert runtimes.default_model_label() == "comparator-gemma-4-31b"


def test_lm_studio_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIVIC_SLM_LM_STUDIO_URL", raising=False)
    assert runtimes.lm_studio_url() == "http://127.0.0.1:1234"


@pytest.mark.parametrize(
    "var",
    [
        "CIVIC_SLM_CANDIDATE_URL",
        "CIVIC_SLM_CANDIDATE_MODEL",
        "CIVIC_SLM_TEACHER_URL",
        "CIVIC_SLM_TEACHER_MODEL",
        "CIVIC_SLM_LOCAL_LLM_URL",
        "CIVIC_SLM_LOCAL_LLM_MODEL",
        "CIVIC_SLM_GEMMA_MODEL",
        "CIVIC_SLM_CIVIC_MODEL",
    ],
)
def test_deprecated_env_raises(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    # Strip every deprecated var first, then set just one; the assertion fires
    # because of *that* one and the message names it.
    for v in (
        "CIVIC_SLM_CANDIDATE_URL",
        "CIVIC_SLM_CANDIDATE_MODEL",
        "CIVIC_SLM_TEACHER_URL",
        "CIVIC_SLM_TEACHER_MODEL",
        "CIVIC_SLM_LOCAL_LLM_URL",
        "CIVIC_SLM_LOCAL_LLM_MODEL",
        "CIVIC_SLM_GEMMA_MODEL",
        "CIVIC_SLM_CIVIC_MODEL",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "anything")
    with pytest.raises(RuntimeError, match=var):
        runtimes.assert_no_deprecated_env()


def test_deprecated_env_silent_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in (
        "CIVIC_SLM_CANDIDATE_URL",
        "CIVIC_SLM_CANDIDATE_MODEL",
        "CIVIC_SLM_TEACHER_URL",
        "CIVIC_SLM_TEACHER_MODEL",
        "CIVIC_SLM_LOCAL_LLM_URL",
        "CIVIC_SLM_LOCAL_LLM_MODEL",
        "CIVIC_SLM_GEMMA_MODEL",
        "CIVIC_SLM_CIVIC_MODEL",
    ):
        monkeypatch.delenv(v, raising=False)
    runtimes.assert_no_deprecated_env()  # no exception
