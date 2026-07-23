"""ChatClient response-parsing edge cases.

Reasoning models served by mlx_lm.server can exhaust the whole `max_tokens`
budget inside hidden chain-of-thought; the returned message then carries a
`reasoning` key and *no* `content` key at all. The client must surface an
actionable error instead of a bare KeyError (which took down every example
of the first side_by_side run).
"""

from __future__ import annotations

from typing import Any

import pytest

from civic_slm.serve.client import ChatClient, MissingContentError


class _CannedResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload


class _CannedHTTPClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _CannedHTTPClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> _CannedResponse:
        return _CannedResponse(self._payload)


def _patch_http(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    def _factory(**_kw: Any) -> _CannedHTTPClient:
        return _CannedHTTPClient(payload)

    monkeypatch.setattr("civic_slm.serve.client.httpx.Client", _factory)


def test_chat_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "a CUP is..."},
                }
            ]
        },
    )
    client = ChatClient(base_url="http://test", model="m")
    assert client.chat("sys", "user").text == "a CUP is..."


def test_chat_raises_actionable_error_when_content_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning-only message (no `content` key) → MissingContentError, not KeyError."""
    _patch_http(
        monkeypatch,
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "reasoning": "Thinking..."},
                }
            ]
        },
    )
    client = ChatClient(base_url="http://test", model="m")
    with pytest.raises(MissingContentError, match="enable_thinking"):
        client.chat("sys", "user")
