"""`civic-slm rag` subapp — local single-jurisdiction RAG over a trained model.

Three commands:

    civic-slm rag index <slug>
        Build the embedding index from data/processed/<slug>.jsonl. Writes
        embeddings.npy + index.jsonl under artifacts/<slug>-rag/.

    civic-slm rag ask <slug> "<question>"
        Retrieve top-k chunks, prepend as a Context: block, send to a
        local mlx_lm.server serving the slug's trained v1 model, print
        the answer and citations.

    civic-slm rag serve <slug>
        Start an OpenAI-compatible HTTP shim that retrieves on every chat
        request and forwards the augmented prompt to mlx_lm.server. Lets
        the existing assistant-ui playground (or any OpenAI client) talk
        to the RAG'd model without code changes.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from civic_slm.config import settings
from civic_slm.logging import configure, get_logger
from civic_slm.serve import models, runtimes
from civic_slm.serve.client import ChatClient
from civic_slm.serve.openai_compat import chat_completions_url, models_url
from civic_slm.serve.rag.index import build_index
from civic_slm.serve.rag.retrieve import format_context, top_k

log = get_logger(__name__)

app = typer.Typer(
    help="Local single-jurisdiction retrieval-augmented inference.", no_args_is_help=True
)


def _index_dir(slug: str) -> Path:
    return settings().artifacts_dir / f"{slug}-rag"


def _resolve_v1_model(slug: str) -> models.Model:
    """Per-jurisdiction model resolution: `<slug>-v1` from the registry or a
    fallback that points at the local artifact dir directly.

    The fallback exists because users can run `civic-slm train jurisdiction`
    against any slug and end up with `artifacts/<slug>-v1-fused/` on disk
    without ever editing the registry. That's the supported path; the
    registry is for stable names people share across machines.
    """
    label = f"{slug}-v1"
    artifact_dir = settings().artifacts_dir / f"{label}-fused"
    if artifact_dir.exists():
        return models.Model(label=label, served_name=str(artifact_dir))
    return models.resolve(label)


@app.command("index")
def index(
    slug: str = typer.Argument(..., help="Jurisdiction slug."),
    embedding_model: str = typer.Option(
        "BAAI/bge-large-en-v1.5",
        "--embedding-model",
        help="HF id of the sentence-transformers encoder.",
    ),
) -> None:
    """Build the embedding index for one jurisdiction."""
    configure()
    out_dir = _index_dir(slug)
    npy, jsonl = build_index(
        slug,
        data_dir=settings().data_dir,
        out_dir=out_dir,
        embedding_model=embedding_model,
    )
    typer.echo(f"Wrote {npy}\nWrote {jsonl}")


@app.command("ask")
def ask(
    slug: str = typer.Argument(..., help="Jurisdiction slug (must have a built index)."),
    question: str = typer.Argument(..., help="Question to ask the model."),
    base_url: str = typer.Option(
        None,
        "--base-url",
        help=(
            "OpenAI-compatible URL. Defaults to $CIVIC_SLM_LM_STUDIO_URL. "
            "Start `mlx_lm.server --model artifacts/<slug>-v1-fused` (or LM "
            "Studio with the v1 model loaded) before calling this."
        ),
    ),
    k: int = typer.Option(4, "--k", help="Top-k retrieved chunks to use as context."),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Per-request max tokens."),
) -> None:
    """Retrieve + answer one question against the slug's trained model."""
    configure()
    base_url = base_url or runtimes.lm_studio_url()
    index_dir = _index_dir(slug)
    results = top_k(question, index_dir=index_dir, k=k)
    context = format_context(results)

    resolved = _resolve_v1_model(slug)
    client = ChatClient(
        base_url=base_url,
        model=resolved.served_name,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": False},
    )
    system = (
        "You are a civic-information assistant. Answer the user's question using "
        "ONLY the passages in the Context: block. Cite sources by their [N] tag. "
        "If the answer is not in the context, say so."
    )
    user = f"{context}\n\nQuestion: {question}"
    response = client.chat(system, user)

    typer.echo(response.text)
    typer.echo("\n--- citations ---")
    for i, r in enumerate(results, 1):
        rec = r.record
        typer.echo(
            f"[{i}] score={r.score:.3f} meeting={rec.meeting_date or 'unknown'} "
            f"source={rec.source_url or rec.doc_id}"
        )


@app.command("serve")
def serve(
    slug: str = typer.Argument(..., help="Jurisdiction slug (must have a built index)."),
    backend_url: str = typer.Option(
        None,
        "--backend-url",
        help=(
            "URL of the model server to forward to (LM Studio or mlx_lm.server). "
            "Defaults to $CIVIC_SLM_LM_STUDIO_URL."
        ),
    ),
    port: int = typer.Option(8767, "--port", help="Port to listen on for the RAG shim."),
    k: int = typer.Option(4, "--k", help="Top-k retrieved chunks per request."),
) -> None:
    """Start the RAG HTTP shim.

    The shim listens for OpenAI-shape `/v1/chat/completions` requests,
    retrieves top-k chunks against the last user message, prepends a
    `Context:` block as a system message, and forwards to the backend
    model server. The existing assistant-ui playground works against
    this URL without modification.
    """
    configure()
    try:
        import uvicorn  # type: ignore[import-not-found]
        from fastapi import FastAPI, Request  # type: ignore[import-not-found]
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
    except ImportError as exc:
        raise typer.BadParameter(
            "`civic-slm rag serve` needs FastAPI + uvicorn. Install with: "
            "uv pip install fastapi uvicorn"
        ) from exc

    backend_url = backend_url or runtimes.lm_studio_url()
    index_dir = _index_dir(slug)
    backend_chat_url = chat_completions_url(backend_url)
    backend_models_url = models_url(backend_url)

    # Pooled httpx clients: a fresh AsyncClient per request reconnects each
    # time and burns through ephemeral ports under load. Keep one chat client
    # (long timeout) for /v1/chat/completions and one short-timeout client
    # for /v1/models.
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:  # pyright: ignore[reportUnknownParameterType]
        app.state.chat_client = httpx.AsyncClient(timeout=600.0)  # pyright: ignore[reportUnknownMemberType]
        app.state.models_client = httpx.AsyncClient(timeout=10.0)  # pyright: ignore[reportUnknownMemberType]
        try:
            yield
        finally:
            await app.state.chat_client.aclose()  # pyright: ignore[reportUnknownMemberType]
            await app.state.models_client.aclose()  # pyright: ignore[reportUnknownMemberType]

    server_app = FastAPI(
        title=f"civic-slm rag shim — {slug}",
        description=(
            "Local single-jurisdiction RAG over `mlx_lm.server`. Bind to "
            "127.0.0.1 only; no auth, no multi-tenant."
        ),
        lifespan=_lifespan,
    )

    # FastAPI's decorator types are inferred lazily; pyright can't see
    # them on a strict pass. The handlers are still type-correct in their
    # bodies, which is what matters for review.
    @server_app.post("/v1/chat/completions")  # pyright: ignore[reportUntypedFunctionDecorator]
    async def chat(req: Request) -> JSONResponse:  # pyright: ignore[reportUnknownParameterType, reportUnusedFunction]
        body = await req.json()
        messages = body.get("messages") or []
        # Last user message drives retrieval.
        question = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                question = str(m.get("content", ""))
                break
        results = top_k(question, index_dir=index_dir, k=k) if question else []
        context_msg = {"role": "system", "content": format_context(results)}
        body["messages"] = [context_msg, *messages]
        client: httpx.AsyncClient = req.app.state.chat_client  # pyright: ignore[reportUnknownMemberType]
        r = await client.post(backend_chat_url, json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    @server_app.get("/v1/models")  # pyright: ignore[reportUntypedFunctionDecorator]
    async def list_models(req: Request) -> dict[str, object]:  # pyright: ignore[reportUnknownParameterType, reportUnusedFunction]
        # Forward whatever the backend reports — the playground uses this.
        client: httpx.AsyncClient = req.app.state.models_client  # pyright: ignore[reportUnknownMemberType]
        r = await client.get(backend_models_url)
        return r.json() if r.status_code == 200 else {"object": "list", "data": []}

    log.info("rag_serve_start", slug=slug, port=port, backend=backend_url)
    uvicorn.run(server_app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    app()
