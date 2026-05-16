"""Transient-HTTP retry wrapper around ChatClient used by the eval runner."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from civic_slm.eval.runner import _chat_with_retry  # pyright: ignore[reportPrivateUsage]
from civic_slm.serve.client import ChatResponse


class _FakeClient:
    """Minimal stand-in for ChatClient that surfaces a scripted sequence."""

    def __init__(self, sequence: list[Any]) -> None:
        self._sequence = list(sequence)
        self.calls = 0

    def chat(self, system: str, user: str) -> ChatResponse:
        self.calls += 1
        item = self._sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _http_status(code: int) -> httpx.HTTPStatusError:
    """Build an HTTPStatusError carrying a real response with the given code."""
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_retry_succeeds_after_transient_error() -> None:
    ok = ChatResponse(text="ok", latency_ms=1.0)
    client = _FakeClient([_http_status(503), ok])
    resp = _chat_with_retry(client, "sys", "user", max_attempts=3)  # type: ignore[arg-type]
    assert resp.text == "ok"
    assert client.calls == 2


def test_retry_raises_after_all_attempts_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip the sleep so tests stay fast.
    def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("civic_slm.eval.runner.time.sleep", _no_sleep)
    client = _FakeClient([_http_status(429), _http_status(429), _http_status(429)])
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        _chat_with_retry(client, "sys", "user", max_attempts=3)  # type: ignore[arg-type]
    assert excinfo.value.response.status_code == 429
    assert client.calls == 3


def test_retry_does_not_retry_404() -> None:
    """A non-transient 4xx (e.g. model-not-loaded) should fail fast on attempt 1."""
    client = _FakeClient([_http_status(404)])
    with pytest.raises(httpx.HTTPStatusError):
        _chat_with_retry(client, "sys", "user", max_attempts=3)  # type: ignore[arg-type]
    assert client.calls == 1


def test_retry_does_not_retry_connect_error() -> None:
    """ConnectError = server not running. Retrying just delays the inevitable."""
    err = httpx.ConnectError("connection refused")
    client = _FakeClient([err])
    with pytest.raises(httpx.ConnectError):
        _chat_with_retry(client, "sys", "user", max_attempts=3)  # type: ignore[arg-type]
    assert client.calls == 1


def test_retry_retries_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("civic_slm.eval.runner.time.sleep", _no_sleep)
    ok = ChatResponse(text="ok", latency_ms=1.0)
    client = _FakeClient([httpx.ReadTimeout("slow"), ok])
    resp = _chat_with_retry(client, "sys", "user", max_attempts=3)  # type: ignore[arg-type]
    assert resp.text == "ok"
    assert client.calls == 2


def test_retry_rejects_zero_attempts() -> None:
    client = _FakeClient([])
    with pytest.raises(ValueError, match="max_attempts"):
        _chat_with_retry(client, "sys", "user", max_attempts=0)  # type: ignore[arg-type]
