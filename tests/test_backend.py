"""Backend dispatch + LocalBackend HTTP shape tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from civic_slm.llm.backend import (
    AnthropicBackend,
    LocalBackend,
    select_backend,
)


def test_select_backend_default_is_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIVIC_SLM_LLM_BACKEND", raising=False)
    backend = select_backend()
    assert isinstance(backend, AnthropicBackend)


def test_select_backend_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "local")
    monkeypatch.setenv("CIVIC_SLM_LOCAL_LLM_URL", "http://example:9999")
    monkeypatch.setenv("CIVIC_SLM_LOCAL_LLM_MODEL", "qwen2.5-72b")
    backend = select_backend()
    assert isinstance(backend, LocalBackend)
    assert backend.base_url == "http://example:9999"
    assert backend.model == "qwen2.5-72b"


def test_select_backend_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "bogus")
    with pytest.raises(ValueError, match="bogus"):
        select_backend()


@dataclass
class _Captured:
    url: str
    payload: dict[str, object]


async def test_local_backend_posts_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[_Captured] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(_Captured(url=str(request.url), payload=json.loads(request.content)))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        )

    transport = httpx.MockTransport(handler)
    real_async = httpx.AsyncClient

    def patched_async(**kwargs: object) -> httpx.AsyncClient:
        return real_async(transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", patched_async)

    backend = LocalBackend(base_url="http://localhost:8081", model="qwen-test")
    out = await backend.complete(system="you are helpful", user="hi", max_tokens=10)
    assert out == "hello"

    assert len(captured) == 1
    assert captured[0].url.endswith("/v1/chat/completions")
    payload = captured[0].payload
    assert payload["model"] == "qwen-test"
    assert payload["max_tokens"] == 10
    msgs = payload["messages"]
    assert isinstance(msgs, list) and msgs[0] == {"role": "system", "content": "you are helpful"}
    assert msgs[1] == {"role": "user", "content": "hi"}
