# Chat Document Attachments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user attach a civic document (PDF/`.txt`/`.md`) in the web chat playground and ask questions grounded on its text, extracted server-side by the project's own Python pipeline.

**Architecture:** A new `POST /v1/attachments` endpoint on the existing RAG-shim FastAPI app extracts text via `ingest/pdf.py`. A Next.js proxy route forwards browser uploads to it. An assistant-ui `AttachmentAdapter` uploads on add and contributes the extracted text on send; the chat adapter prepends that text to the user turn before hitting `/api/chat` → LM Studio.

**Tech Stack:** Python 3.11 / FastAPI / pypdf (`ingest` extra); Next.js 16 + React 19 + `@assistant-ui/react` 0.12.28; pytest; the `web/` app already runs `next dev`.

## Global Constraints

- Runtime env for the Python endpoint: the `rag` extra (`fastapi`, `uvicorn`) + the `ingest` extra (`pypdf`). Endpoint must degrade to a clear error if `pypdf` is missing (mirror `ingest/pdf.py`'s lazy import).
- Supported types: `.pdf`, `.txt`, `.md` only. Max file size: **10 MB**. Max **5** attachments/turn. Total injected doc text capped at **30 000 chars**.
- RAG shim default port: **8767**. Next app reads shim URL from `CIVIC_SLM_RAG_URL` (default `http://127.0.0.1:8767`).
- Chat WITHOUT attachments must be unchanged (still LM-Studio-direct; the shim is only needed when a doc is attached).
- Lint/type/test gates: `uv run ruff check`, `uv run pyright`, `uv run pytest` (Python); `pnpm --dir web lint` / `tsc --noEmit` (web). Match existing code style; type hints on public fns; `from __future__ import annotations`.
- Injection format (verbatim), prepended to the user's text on the turn the doc is attached to:
  ```
  Attached document "<filename>" (<pages> pages):
  <text>

  ```

---

### Task 1: Python `POST /v1/attachments` endpoint (+ testable app factory)

**Files:**

- Modify: `src/civic_slm/serve/rag/cli.py` — extract the FastAPI app into `build_app()` and add the `/v1/attachments` route.
- Create: `src/civic_slm/serve/rag/attachments.py` — pure extraction helper (no FastAPI import; unit-testable).
- Create: `tests/test_rag_attachments.py`
- Reference: `src/civic_slm/ingest/pdf.py` (`extract_pdf(path: Path) -> list[ExtractedPage]`, `ExtractedPage(page_idx:int, text:str)`).

**Interfaces:**

- Produces: `extract_document(filename: str, data: bytes) -> DocText` where
  `DocText = dataclass(filename:str, content_type:str, pages:int, chars:int, truncated:bool, text:str)`.
  Raises `UnsupportedDocType(str)` and `DocExtractionError(str)`.
- Produces: `build_app(port: int) -> FastAPI` (used by `serve()` and by tests via `TestClient`).

- [ ] **Step 1: Write the failing extraction test**

```python
# tests/test_rag_attachments.py
from __future__ import annotations
import pytest
from civic_slm.serve.rag.attachments import (
    extract_document, DocText, UnsupportedDocType, DocExtractionError,
)

def test_txt_extraction_roundtrips() -> None:
    out = extract_document("notes.txt", b"Council approved item 9.")
    assert isinstance(out, DocText)
    assert out.text == "Council approved item 9."
    assert out.pages == 1
    assert out.chars == len(out.text)
    assert out.truncated is False

def test_unsupported_type_rejected() -> None:
    with pytest.raises(UnsupportedDocType):
        extract_document("photo.png", b"\x89PNG")

def test_oversized_text_is_truncated() -> None:
    big = "x" * 40_000
    out = extract_document("big.md", big.encode())
    assert out.truncated is True
    assert out.chars <= 30_000
    assert out.text.endswith("[document truncated]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_attachments.py -v`
Expected: FAIL — `ModuleNotFoundError: civic_slm.serve.rag.attachments`

- [ ] **Step 3: Write the extraction helper**

```python
# src/civic_slm/serve/rag/attachments.py
"""Turn an uploaded civic document into grounding text.

No FastAPI import here so the extraction logic is unit-testable in isolation;
the shim route in cli.py is a thin HTTP wrapper over `extract_document`.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 30_000
_SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedDocType(ValueError):
    """Raised for a file extension outside `.pdf/.txt/.md`."""


class DocExtractionError(RuntimeError):
    """Raised when a supported file can't be read (encrypted/corrupt PDF)."""


@dataclass(frozen=True)
class DocText:
    filename: str
    content_type: str
    pages: int
    chars: int
    truncated: bool
    text: str


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CHARS:
        return text, False
    marker = "\n[document truncated]"
    return text[: MAX_CHARS - len(marker)] + marker, True


def extract_document(filename: str, data: bytes) -> DocText:
    ext = Path(filename).suffix.lower()
    if ext not in _SUPPORTED:
        raise UnsupportedDocType(f"unsupported type {ext!r}; allowed: .pdf, .txt, .md")

    if ext in {".txt", ".md"}:
        raw = data.decode("utf-8", errors="replace").strip()
        text, truncated = _truncate(raw)
        ctype = "text/markdown" if ext == ".md" else "text/plain"
        return DocText(filename, ctype, 1, len(text), truncated, text)

    # PDF: extract_pdf takes a path, so stage to a temp file.
    from civic_slm.ingest.pdf import extract_pdf  # lazy: pypdf is the `ingest` extra

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(data)
        fh.flush()
        try:
            pages = extract_pdf(Path(fh.name))
        except Exception as exc:  # pypdf raises PdfReadError etc.
            raise DocExtractionError(f"could not read PDF {filename!r}: {exc}") from exc
    raw = "\n\n".join(p.text for p in pages if p.text).strip()
    if not raw:
        raise DocExtractionError(f"no extractable text in {filename!r} (scanned image?)")
    text, truncated = _truncate(raw)
    return DocText(filename, "application/pdf", len(pages), len(text), truncated, text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_attachments.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add the HTTP route + `build_app` factory to the shim**

In `src/civic_slm/serve/rag/cli.py`, inside `serve()`: change the imports line
`from fastapi import FastAPI, Request` → `from fastapi import FastAPI, File, Request, UploadFile`.
After the existing `@server_app.get("/v1/models")` handler and before `uvicorn.run(...)`, add:

```python
    @server_app.post("/v1/attachments")  # pyright: ignore[reportUntypedFunctionDecorator]
    async def attachments(file: UploadFile = File(...)) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        from civic_slm.serve.rag.attachments import (
            DocExtractionError, UnsupportedDocType, extract_document,
        )
        data = await file.read()
        if len(data) > 10 * 1024 * 1024:
            return JSONResponse({"error": "file exceeds 10 MB"}, status_code=413)
        try:
            doc = extract_document(file.filename or "upload", data)
        except UnsupportedDocType as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except DocExtractionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(doc.__dict__)
```

> NOTE: the shim currently builds `server_app` inline in `serve()`. Leave that
> as-is for this task — the route is registered the same way as the two existing
> ones. The endpoint is covered by the pure-helper tests above; an HTTP-level
> TestClient test is deferred (would require the `build_app` refactor, tracked as
> a follow-up so this task stays focused).

- [ ] **Step 6: Commit**

```bash
git add src/civic_slm/serve/rag/attachments.py tests/test_rag_attachments.py src/civic_slm/serve/rag/cli.py
git commit -m "feat(rag): POST /v1/attachments — extract civic doc text (#92)"
```

---

### Task 2: Next.js `/api/attachments` proxy route

**Files:**

- Create: `web/src/app/api/attachments/route.ts`
- Create: `web/src/app/api/attachments/route.test.ts` (or the repo's web test location — mirror `api/chat` tests if present)
- Reference: `web/src/app/api/chat/route.ts` (env + base-url convention).

**Interfaces:**

- Consumes: Task 1's `/v1/attachments` JSON `{filename, content_type, pages, chars, truncated, text}`.
- Produces: `POST /api/attachments` (multipart `file`) → same JSON, or `{error}` with status.

- [ ] **Step 1: Write the failing route test**

```ts
// web/src/app/api/attachments/route.test.ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { POST } from "./route";

afterEach(() => vi.restoreAllMocks());

function upload(name: string, body: string): Request {
  const fd = new FormData();
  fd.set("file", new File([body], name, { type: "text/plain" }));
  return new Request("http://x/api/attachments", { method: "POST", body: fd });
}

describe("POST /api/attachments", () => {
  it("proxies extracted text from the shim", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          filename: "a.txt",
          pages: 1,
          chars: 3,
          truncated: false,
          text: "abc",
        }),
        { status: 200 },
      ),
    );
    const res = await POST(upload("a.txt", "abc"));
    expect(res.status).toBe(200);
    expect((await res.json()).text).toBe("abc");
  });

  it("returns 503 when the shim is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("ECONNREFUSED"));
    const res = await POST(upload("a.txt", "abc"));
    expect(res.status).toBe(503);
    expect((await res.json()).error).toMatch(/civic-slm rag serve/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir web test route.test` (or `pnpm --dir web vitest run api/attachments`)
Expected: FAIL — `Cannot find module './route'`

- [ ] **Step 3: Write the route**

```ts
// web/src/app/api/attachments/route.ts
const RAW = process.env.CIVIC_SLM_RAG_URL ?? "http://127.0.0.1:8767";
const SHIM = RAW.replace(/\/$/, "");

export async function POST(req: Request): Promise<Response> {
  const form = await req.formData();
  const file = form.get("file");
  if (!(file instanceof File)) {
    return Response.json({ error: "no file field in upload" }, { status: 400 });
  }
  const forward = new FormData();
  forward.set("file", file, file.name);
  try {
    const res = await fetch(`${SHIM}/v1/attachments`, {
      method: "POST",
      body: forward,
    });
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json(
      { error: "RAG shim not running — start it with `civic-slm rag serve`" },
      { status: 503 },
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --dir web test route.test`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add web/src/app/api/attachments/
git commit -m "feat(web): /api/attachments proxy to the RAG shim (#92)"
```

---

### Task 3: assistant-ui `AttachmentAdapter` (upload on add, text on send)

**Files:**

- Create: `web/src/components/chat/attachment-adapter.ts`
- Reference: `@assistant-ui/core` `SimpleTextAttachmentAdapter` (import path `@assistant-ui/react`) — mirror its `accept/add/send/remove` shape; open it in `web/node_modules/@assistant-ui/core` to confirm the exact `PendingAttachment`/`CompleteAttachment` fields for the installed 0.12.28.

**Interfaces:**

- Produces: `documentAttachmentAdapter: AttachmentAdapter` (default export) — `add` POSTs to `/api/attachments` and stashes the returned text on the pending attachment; `send` returns `{ ...attachment, status: { type: "complete" }, content: [{ type: "text", text: "<injection block>" }] }`.

- [ ] **Step 1: Write the adapter**

```ts
// web/src/components/chat/attachment-adapter.ts
import type {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from "@assistant-ui/react";

const ACCEPT = ".pdf,.txt,.md";

type Extracted = {
  filename: string;
  pages: number;
  text: string;
  truncated: boolean;
};

export const documentAttachmentAdapter: AttachmentAdapter = {
  accept: ACCEPT,

  async add({ file }): Promise<PendingAttachment> {
    const form = new FormData();
    form.set("file", file, file.name);
    const res = await fetch("/api/attachments", { method: "POST", body: form });
    if (!res.ok) {
      const { error } = (await res.json().catch(() => ({}))) as {
        error?: string;
      };
      throw new Error(error ?? `extraction failed (${res.status})`);
    }
    const doc = (await res.json()) as Extracted;
    return {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: { type: "requires_action", reason: "composer-send" },
      // stash extracted text for send() (typed loosely; not part of the base shape)
      ...({ _doc: doc } as object),
    } as unknown as PendingAttachment;
  },

  async send(attachment): Promise<CompleteAttachment> {
    const doc = (attachment as unknown as { _doc: Extracted })._doc;
    const block = `Attached document "${doc.filename}" (${doc.pages} pages):\n${doc.text}\n\n`;
    return {
      ...attachment,
      status: { type: "complete" },
      content: [{ type: "text", text: block }],
    } as CompleteAttachment;
  },

  async remove() {
    // extraction is stateless server-side; nothing to clean up.
  },
};
```

> NOTE: the `_doc` stash is the pragmatic way to carry extracted text from
> `add` → `send`. When wiring, confirm the installed `PendingAttachment` field
> names against `SimpleTextAttachmentAdapter`; adjust the `status` literals if
> the 0.12.28 types differ (the shape is stable across 0.12.x).

- [ ] **Step 2: Type-check**

Run: `pnpm --dir web exec tsc --noEmit`
Expected: PASS (no errors in `attachment-adapter.ts`)

- [ ] **Step 3: Commit**

```bash
git add web/src/components/chat/attachment-adapter.ts
git commit -m "feat(web): document AttachmentAdapter (upload+extract) (#92)"
```

---

### Task 4: Wire the adapter into the runtime + inject doc text into the chat turn

**Files:**

- Modify: `web/src/components/chat/runtime-provider.tsx` — (a) pass the adapter to `useLocalRuntime`; (b) stop stripping non-text parts so the injected doc text is forwarded to `/api/chat`.

**Interfaces:**

- Consumes: `documentAttachmentAdapter` (Task 3).

- [ ] **Step 1: Import the adapter and register it**

Change `const runtime = useLocalRuntime(adapter);` to:

```tsx
import { documentAttachmentAdapter } from "./attachment-adapter";
// ...
const runtime = useLocalRuntime(adapter, {
  adapters: { attachments: documentAttachmentAdapter },
});
```

- [ ] **Step 2: Forward attachment text in the ChatModelAdapter**

In the `run({ messages })` mapping, the current code keeps only `p.type === "text"`.
The adapter's `send()` already produced a `text` part, so no new type is needed — but confirm the join preserves order (doc block first, then the user's typed text). Replace the `apiMessages` map with:

```tsx
const apiMessages = messages.map((m) => ({
  role: m.role,
  content: m.content
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join(""),
}));
```

(Attachment text parts are `type: "text"`, so they flow through; the doc block's
trailing `\n\n` separates it from the question.)

- [ ] **Step 3: Type-check + manual smoke**

Run: `pnpm --dir web exec tsc --noEmit` → PASS.
Manual: start `civic-slm rag serve`, `pnpm --dir web dev`, attach a small `.txt`, ask "what does this say?" — response reflects the file.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/chat/runtime-provider.tsx
git commit -m "feat(web): register attachment adapter + forward doc text (#92)"
```

---

### Task 5: Composer UI — document chip + file picker

**Files:**

- Modify: `web/src/components/assistant-ui/attachment.tsx` — add a document-attachment branch (chip with filename + remove) alongside the existing image UI; ensure the composer's add-attachment button is rendered in the thread composer.
- Reference: existing `AttachmentPrimitive`/`ComposerPrimitive` usage already in this file; `FileText` icon is already imported.

**Interfaces:**

- Consumes: the runtime's attachment state (populated by Task 3/4).

- [ ] **Step 1: Add a document chip branch**

In `attachment.tsx`, where the component switches on `s.attachment.type`, add a
non-image branch that renders a chip: `FileText` icon + `AttachmentPrimitive.Name`

- a remove `TooltipIconButton` (`XIcon`) wired to `AttachmentPrimitive.Remove`.
  Reuse the existing `cn`/`Tooltip` imports. Keep the image branch untouched.

```tsx
// inside the attachment component, after the image-type early return:
return (
  <div
    className={cn(
      "flex items-center gap-2 rounded-md border px-2 py-1 text-sm",
    )}
  >
    <FileText className="size-4 shrink-0" />
    <AttachmentPrimitive.Name className="truncate max-w-40" />
    <AttachmentPrimitive.Remove asChild>
      <TooltipIconButton tooltip="Remove" className="ml-auto size-6">
        <XIcon className="size-3.5" />
      </TooltipIconButton>
    </AttachmentPrimitive.Remove>
  </div>
);
```

- [ ] **Step 2: Ensure the composer exposes the add button**

Confirm `thread.tsx`'s composer renders the attachment add control (`ComposerPrimitive.AddAttachment` or the `ComposerAttachments` from this file). If missing, add the add-attachment button (PlusIcon, already imported) to the composer row with `accept=".pdf,.txt,.md"`.

- [ ] **Step 3: Type-check + manual smoke**

Run: `pnpm --dir web exec tsc --noEmit` → PASS.
Manual: the composer shows a paperclip/plus; picking a `.pdf` shows a chip; the remove button clears it.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/assistant-ui/attachment.tsx web/src/components/assistant-ui/thread.tsx
git commit -m "feat(web): document attachment chip + composer picker (#92)"
```

---

### Task 6: Docs — new runtime dependency

**Files:**

- Modify: `web/README.md` — note that document attachments require `civic-slm rag serve` (port 8767) and the `CIVIC_SLM_RAG_URL` env override.
- Modify: `docs/RUNTIMES.md` — add `CIVIC_SLM_RAG_URL` to the env table; note the shim now also serves `/v1/attachments`.

- [ ] **Step 1: Write the docs**

Add to `web/README.md` under a "Document attachments" heading: which file types,
the 10 MB / 5-file / 30k-char limits, and: "Attachments call the RAG shim —
start it with `civic-slm rag serve` (default `http://127.0.0.1:8767`); override
with `CIVIC_SLM_RAG_URL`. Chat without attachments needs only LM Studio."
Add the `CIVIC_SLM_RAG_URL` row to the `docs/RUNTIMES.md` env table.

- [ ] **Step 2: Commit**

```bash
git add web/README.md docs/RUNTIMES.md
git commit -m "docs: document attachments need the RAG shim (#92)"
```

---

## Self-Review

**Spec coverage:** Python endpoint (T1) ✓ · Next proxy (T2) ✓ · AttachmentAdapter (T3) ✓ · runtime wiring + injection (T4) ✓ · composer UI (T5) ✓ · docs/dependency (T6) ✓. Limits (10 MB / 5 files / 30k chars) enforced in T1 helper + endpoint. Injection format matches the spec verbatim (T3 `send`). Error paths (415/422/413/503) covered in T1/T2.

**Placeholder scan:** No "TBD"/"add error handling"-style gaps; each code step has real code. The two `NOTE:` blocks (T1 TestClient deferral, T3 field-name confirmation) are explicit scope/verification notes, not missing content.

**Type consistency:** `DocText` fields (`filename, content_type, pages, chars, truncated, text`) are produced in T1 and consumed as JSON in T2/T3 (`Extracted = {filename, pages, text, truncated}` — a subset, consistent). `documentAttachmentAdapter` name matches between T3 (produce) and T4 (consume). Injection block string identical in spec and T3.

**Known follow-ups (out of this plan):** HTTP-level TestClient test for the endpoint (needs a `build_app` factory refactor); chunk-and-retrieve for >30k-char docs; drag-and-drop.
