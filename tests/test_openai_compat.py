"""Shared `/v1/chat/completions` URL plumbing."""

from __future__ import annotations

import pytest

from civic_slm.serve.openai_compat import chat_completions_url, models_url


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:1234",
        "http://127.0.0.1:1234/",
        "http://127.0.0.1:1234/v1",
        "http://127.0.0.1:1234/v1/",
        "http://127.0.0.1:1234/v1/v1",  # the pathological double-append
    ],
)
def test_chat_completions_url_collapses_trailing_v1(base: str) -> None:
    assert chat_completions_url(base) == "http://127.0.0.1:1234/v1/chat/completions"


@pytest.mark.parametrize(
    "base",
    [
        "http://host:8080",
        "http://host:8080/v1",
        "http://host:8080/v1/",
    ],
)
def test_models_url(base: str) -> None:
    assert models_url(base) == "http://host:8080/v1/models"


@pytest.mark.parametrize("base", ["", "   ", "/"])
def test_empty_base_url_raises_actionable(base: str) -> None:
    """A misconfigured env (CIVIC_SLM_LM_STUDIO_URL='') should fail loud, not
    produce a relative URL that httpx will later reject with a cryptic error."""
    with pytest.raises(ValueError, match="empty"):
        chat_completions_url(base)
