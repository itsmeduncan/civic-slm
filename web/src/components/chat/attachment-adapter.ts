import type {
  AttachmentAdapter,
  PendingAttachment,
  CompleteAttachment,
} from "@assistant-ui/react";

const ACCEPT = ".pdf,.txt,.md";

/**
 * Shape of the JSON body returned by `POST /api/attachments`, which proxies
 * to the Python RAG shim's `/v1/attachments` route (`DocText.__dict__` in
 * `src/civic_slm/serve/rag/attachments.py`). Only the fields the injection
 * block needs are used, but the full response shape is typed for clarity.
 */
interface ExtractedDocument {
  filename: string;
  content_type: string;
  pages: number;
  chars: number;
  truncated: boolean;
  text: string;
}

/** Builds the text block injected into the outgoing message on send(). */
function injectionBlock(doc: ExtractedDocument): string {
  return `Attached document "${doc.filename}" (${doc.pages} pages):\n${doc.text}\n\n`;
}

/**
 * Uploads PDF/TXT/MD attachments to `/api/attachments` for server-side text
 * extraction, then contributes the extracted text as a message content part
 * on send.
 *
 * The extracted text is computed once, in `add()` (where the upload
 * already happens), and carried forward on the attachment's own `content`
 * field — a real, optional field on `BaseAttachment` shared by both
 * `PendingAttachment` and `CompleteAttachment` in `@assistant-ui/core`
 * (confirmed by reading `types/attachment.d.ts` and `SimpleTextAttachmentAdapter`
 * in the installed `@assistant-ui/core@0.1.17`) — rather than a bespoke stash
 * property. `send()` only needs to flip `status` to `complete`.
 */
export const documentAttachmentAdapter: AttachmentAdapter = {
  accept: ACCEPT,

  async add({ file }): Promise<PendingAttachment> {
    const form = new FormData();
    form.set("file", file, file.name);

    const res = await fetch("/api/attachments", {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { error?: string };
      throw new Error(body.error ?? `extraction failed (${res.status})`);
    }
    const doc = (await res.json()) as ExtractedDocument;

    return {
      id: crypto.randomUUID(),
      type: "document",
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: { type: "requires-action", reason: "composer-send" },
      content: [{ type: "text", text: injectionBlock(doc) }],
    };
  },

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    return {
      ...attachment,
      status: { type: "complete" },
      content: attachment.content ?? [],
    };
  },

  async remove() {
    // Extraction is stateless server-side; nothing to clean up.
  },
};
