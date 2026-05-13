"""Thin OpenAI-compatible chat client used by the eval runner.

LM Studio exposes an OpenAI-compatible `/v1/chat/completions` endpoint
on port 1234 by default. Eval doesn't need streaming, tool use, or any
other niceties — just a synchronous chat call with a deadline and a
latency measurement. Keeping this client deliberately small means we can
swap backends without rewriting eval logic.

Timeout: defaults to 600s (10 min). Reasoning models (Qwen 3.6, Gemma 4) can
burn the full `max_tokens` budget on a hidden chain-of-thought before emitting
any visible content — extraction prompts with dense documents routinely take
5-8 minutes per example on a warm Mac, and you want headroom above that.
Override per-instance via `timeout_s=` or globally via
`CIVIC_SLM_TIMEOUT_S=<seconds>` (bump to 1800 for very long-context evals).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx


def _default_timeout() -> float:
    raw = os.environ.get("CIVIC_SLM_TIMEOUT_S", "600")
    try:
        return float(raw)
    except ValueError:
        return 600.0


@dataclass(frozen=True)
class ChatResponse:
    text: str
    latency_ms: float


@dataclass(frozen=True)
class ChatClient:
    """OpenAI-compatible chat client."""

    base_url: str
    model: str
    api_key: str = "not-needed"
    timeout_s: float = field(default_factory=_default_timeout)
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int = 0

    def chat(self, system: str, user: str) -> ChatResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": False,
        }
        # Accept both `http://host:port` and `http://host:port/v1` to match the
        # common OpenAI-SDK convention without producing `/v1/v1/...`.
        root = self.base_url.rstrip("/").removesuffix("/v1")
        url = f"{root}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                # Surface the server's error body — LM Studio includes the
                # actual reason here ("model not loaded", "context overflow",
                # etc.) and httpx's default raise_for_status throws it away.
                raise httpx.HTTPStatusError(
                    f"HTTP {r.status_code} from {url} (model={self.model!r}): {r.text[:500]}",
                    request=r.request,
                    response=r,
                )
            data = r.json()
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = str(data["choices"][0]["message"]["content"])
        return ChatResponse(text=text, latency_ms=latency_ms)
