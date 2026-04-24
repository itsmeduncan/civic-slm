"""Thin OpenAI-compatible chat client used by the eval runner.

Both `mlx_lm.server` and `llama.cpp`'s `llama-server` expose an
OpenAI-compatible `/v1/chat/completions` endpoint. Eval doesn't need streaming,
tool use, or any other niceties — just a synchronous chat call with a deadline
and a latency measurement. Keeping this client deliberately small means we can
swap backends (or add vLLM later) without rewriting eval logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


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
    timeout_s: float = 120.0
    max_tokens: int = 512
    temperature: float = 0.0

    def chat(self, system: str, user: str) -> ChatResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = str(data["choices"][0]["message"]["content"])
        return ChatResponse(text=text, latency_ms=latency_ms)
