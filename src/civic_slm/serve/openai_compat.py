"""Shared URL plumbing for the OpenAI-compatible servers we talk to.

Four call sites build `/v1/chat/completions` URLs slightly differently
today: `ChatClient`, `LocalBackend.complete`, the doctor `_ping_chat`,
and the RAG shim. Three of them strip a trailing `/v1` from a base URL,
one didn't, and one only stripped a single suffix instead of looping.

Putting the URL construction here means: one fix for the trailing-slash
edge cases, one place to add `/v1/embeddings` if we ever need it, no more
"works in eval, breaks in doctor" subtle divergences.

Convention: `base_url` may be any of `http://host`, `http://host:port`,
`http://host:port/`, `http://host:port/v1`, or even an accidentally
doubled `http://host:port/v1/v1`. `_canonical_root` collapses all of
those to `http://host:port`.
"""

from __future__ import annotations


def _canonical_root(base_url: str) -> str:
    """Strip trailing slash(es) and any trailing `/v1` segments.

    A loop, not a single `.removesuffix`, so that pathological inputs like
    `http://host/v1/v1` collapse cleanly. Operates only on the suffix —
    middle-segment `/v1`s (which would be weird but legal in a proxy URL)
    are preserved.

    Raises `ValueError` on empty / whitespace-only input: a downstream
    httpx call against `/v1/chat/completions` with no host would fail with
    a cryptic "Request URL is missing an 'http://' or 'https://' protocol"
    instead of telling the operator their LM Studio URL env var is empty.
    """
    root = base_url.strip().rstrip("/")
    if not root:
        raise ValueError("base_url is empty; set CIVIC_SLM_LM_STUDIO_URL or pass --base-url.")
    while root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def chat_completions_url(base_url: str) -> str:
    """Return `<root>/v1/chat/completions` for the given base URL."""
    return f"{_canonical_root(base_url)}/v1/chat/completions"


def models_url(base_url: str) -> str:
    """Return `<root>/v1/models` for the given base URL."""
    return f"{_canonical_root(base_url)}/v1/models"
