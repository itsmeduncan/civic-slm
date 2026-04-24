"""Single chat-completion backend used by synth + judge + (indirectly) the crawler.

Why centralize: synth, the side-by-side judge, and the browser-use crawler all
need to reach an LLM with a "given a prompt, return text" shape. Without a
shared abstraction each call site picks its own SDK and we end up with three
divergent retry/rate-limit/auth code paths. Here we route on a single env var
so a fully-local Mac install can swap Anthropic out without touching any call
site.

Selection precedence:
  1. Explicit `Backend` instance passed in (tests).
  2. `CIVIC_SLM_LLM_BACKEND` env: `local` | `anthropic`. Default `anthropic` to
     preserve existing behavior.
  3. For `local`: `CIVIC_SLM_LOCAL_LLM_URL` (default `http://127.0.0.1:8081`)
     and `CIVIC_SLM_LOCAL_LLM_MODEL` (default `default` — whatever the server
     reports).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Protocol

import httpx


class Backend(Protocol):
    """Minimal completion shape every call site agrees on."""

    @property
    def model(self) -> str: ...

    async def complete(self, *, system: str | None, user: str, max_tokens: int = 4096) -> str: ...


@dataclass(frozen=True)
class LocalBackend:
    """OpenAI-compatible HTTP client for `mlx_lm.server` or `llama-server`.

    Uses /v1/chat/completions with messages=[{system}, {user}]. Returns the
    assistant message content as plain text.
    """

    base_url: str = "http://127.0.0.1:8081"
    model: str = "default"
    api_key: str = "not-needed"
    timeout_s: float = 600.0

    async def complete(self, *, system: str | None, user: str, max_tokens: int = 4096) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data = r.json()
        return str(data["choices"][0]["message"]["content"])


@dataclass(frozen=True)
class AnthropicBackend:
    """Wraps the official Anthropic SDK (lazy import — keeps it optional)."""

    model: str = "claude-opus-4-7"
    timeout_s: float = 600.0

    async def complete(self, *, system: str | None, user: str, max_tokens: int = 4096) -> str:
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        from civic_slm.config import require

        client = AsyncAnthropic(api_key=require("ANTHROPIC_API_KEY"))
        if system:
            msg = await client.messages.create(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        else:
            msg = await client.messages.create(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user}],
            )
        return "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
        )


def select_backend(*, default_anthropic_model: str = "claude-opus-4-7") -> Backend:
    """Pick a backend from env. See module docstring for precedence."""
    choice = os.environ.get("CIVIC_SLM_LLM_BACKEND", "anthropic").strip().lower()
    if choice == "local":
        return LocalBackend(
            base_url=os.environ.get("CIVIC_SLM_LOCAL_LLM_URL", "http://127.0.0.1:8081"),
            model=os.environ.get("CIVIC_SLM_LOCAL_LLM_MODEL", "default"),
        )
    if choice == "anthropic":
        return AnthropicBackend(model=default_anthropic_model)
    raise ValueError(
        f"CIVIC_SLM_LLM_BACKEND={choice!r} is not recognized. "
        "Set it to 'local' or 'anthropic' (or unset to default to anthropic)."
    )


def complete_sync(
    backend: Backend, *, system: str | None, user: str, max_tokens: int = 4096
) -> str:
    """Convenience wrapper for callers that aren't async (the judge)."""
    return asyncio.run(backend.complete(system=system, user=user, max_tokens=max_tokens))
