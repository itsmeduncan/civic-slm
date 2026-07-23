# Design — Document attachments in the chat playground (#92)

**Status:** approved (2026-07-22) · **Scope:** `web/` dogfooding playground + one Python endpoint

## Overview

Let a user attach a civic document (PDF, `.txt`, `.md`) in the chat composer.
The document's text is extracted **server-side using the project's existing
Python pipeline** (`ingest/pdf.py`) and injected into the chat as grounding
context, so the user can ask questions about it ("summarize this staff report",
"what's the fiscal impact of item 9").

## Goals / Non-goals

**Goals**

- Attach one or more documents in the composer; see them as removable chips.
- Extract text via the battle-tested `extract_pdf()` (same code that builds the
  training corpus), not a JS reimplementation.
- Inject extracted text as context on the turn it's attached to.
- Actionable errors (unreadable PDF, oversize, shim down).

**Non-goals (YAGNI)**

- Images / model vision (E4B vision through LM Studio is fragile — probed and it
  errored on a basic image).
- RAG retrieval / chunk-ranking over attachments — v1 injects full text,
  truncated. (Chunking exists in `chunk_text()`; wire it later if docs routinely
  exceed the budget.)
- Persistence, multi-user, auth — `web/` is explicitly a single-user playground.

## Architecture (5 components)

| #   | Layer              | File                                                     | Change                                                                                                                                                    |
| --- | ------------------ | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Python endpoint    | `src/civic_slm/serve/rag/cli.py`                         | Add `POST /v1/attachments` to the existing RAG-shim FastAPI app.                                                                                          |
| 2   | Next API route     | `web/src/app/api/attachments/route.ts` (new)             | Receive the browser upload, proxy to the shim, return extracted text.                                                                                     |
| 3   | Attachment adapter | `web/src/components/chat/attachment-adapter.ts` (new)    | assistant-ui `AttachmentAdapter`: `add` → upload+extract; `send` → contribute text part.                                                                  |
| 4   | Composer UI        | `web/src/components/assistant-ui/attachment.tsx` (adapt) | Document chip (name/size/remove) + file picker `accept=".pdf,.txt,.md"`.                                                                                  |
| 5   | Chat adapter       | `web/src/components/chat/runtime-provider.tsx`           | Stop stripping non-text parts; include the extracted-doc text in the `/api/chat` message. Wire `useLocalRuntime(adapter, { adapters: { attachments } })`. |

## Data flow

```
composer file-pick
  → AttachmentAdapter.add(file)
      → POST /api/attachments (multipart)            [Next route]
          → POST http://127.0.0.1:8767/v1/attachments [RAG shim]
              → extract_pdf() / read text → { text, pages, chars, truncated }
      ← extracted text stored on the attachment
  → user sends turn
      → AttachmentAdapter.send() contributes a text content part
      → ChatModelAdapter.run() prepends the doc block to the user message
      → POST /api/chat → LM Studio (google/gemma-4-e4b)
```

## API contracts

**`POST /v1/attachments`** (RAG shim, port 8767) — `multipart/form-data`, field `file`.
Response `200`:

```json
{
  "filename": "staff-report.pdf",
  "content_type": "application/pdf",
  "pages": 12,
  "chars": 41833,
  "truncated": false,
  "text": "..."
}
```

Errors: `415` unsupported type; `413` over 10 MB; `422` extraction failed
(unreadable/encrypted PDF) with a human message. `.txt/.md` are decoded UTF-8
(errors="replace"); `.pdf` goes through `extract_pdf()` (pages joined by `\n\n`).

**`POST /api/attachments`** (Next) — same multipart in; proxies to
`CIVIC_SLM_RAG_URL` (default `http://127.0.0.1:8767`). On shim-unreachable →
`503` with `{ "error": "RAG shim not running — start it with `civic-slm rag serve`" }`.

## Key decisions (concrete)

- **Injection format** — the doc text is prepended to the user turn as:
  ```
  Attached document "<filename>" (<pages> pages):
  <text>

  <user's question>
  ```
  Multiple docs stack in attach order.
- **Truncation** — enforced at two layers, both simple truncation (not
  proportional): (1) the RAG shim caps each file's extracted text at
  **30 000 chars** per doc, setting `truncated: true` so the UI can badge it;
  (2) `runtime-provider.tsx` caps the **combined** attachment text across all
  attachments on a turn at **30 000 chars** total (~7.5k tokens; comfortable
  within E4B's context alongside the answer) — if the joined text from every
  attachment exceeds the total, it's sliced to the budget and suffixed with
  `\n[attachments truncated]\n\n`. This guards against several attachments
  each under the per-file cap still blowing the combined budget.
- **Limits** — **10 MB**/file, types `.pdf/.txt/.md`. There is no enforced
  per-turn file-count limit — assistant-ui's `AttachmentAdapter` has no
  `maxCount` hook, so "attach a few documents per turn" is advisory guidance
  in the UI copy, not a hard cap.
- **New runtime dependency** — attachments require `civic-slm rag serve` running.
  Documented in `web/README.md` + `docs/RUNTIMES.md`; the shim-down error is
  actionable. Chat without attachments is unaffected (still LM-Studio-direct).

## Error handling

- Unreadable/encrypted PDF → chip shows an error state, no text injected.
- Oversize / unsupported type → rejected at pick time (client pre-check) and by
  the endpoint (defense in depth).
- Shim unreachable → composer surfaces the "start the shim" hint; the turn can
  still be sent without the attachment.

## Testing

- **Python** (`tests/`): `POST /v1/attachments` with a small fixture PDF → asserts
  non-empty text + page count; a `.txt` upload → decoded text; a bogus file → 422. Reuses the existing MLX-skip-free test style; FastAPI `TestClient`.
- **Next route**: unit test mocking the shim (200 + 503 paths).
- **UI**: attachment-chip render/remove smoke; adapter `add`/`send` contract test.

## Out of scope / future

Chunk-and-retrieve for very large docs (reuse `chunk_text()` + the RAG index);
image/vision once E4B vision is verified through the serving layer; drag-and-drop.
