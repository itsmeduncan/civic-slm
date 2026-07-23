"use client";

import { useMemo } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { documentAttachmentAdapter } from "./attachment-adapter";
import type { PromptKey } from "./types";
import { SYSTEM_PROMPTS } from "./types";

// Endpoint-side (`/v1/attachments`) already caps each file at 30k chars, but
// that's a per-doc guard — a turn with several attachments can still exceed
// the budget in aggregate. This is the total cap across all attachments on
// one turn, enforced client-side before the message is sent.
const MAX_ATTACHMENT_CHARS = 30_000;

export function ChatRuntimeProvider({
  selectedModel,
  activePrompt,
  temperature,
  maxTokens,
  children,
}: {
  selectedModel: string;
  activePrompt: PromptKey;
  temperature: number;
  maxTokens: number;
  children: React.ReactNode;
}) {
  const adapter = useMemo<ChatModelAdapter>(
    () => ({
      async *run({ messages, abortSignal }) {
        // Attachment text (injected by documentAttachmentAdapter.add()) lives
        // on `m.attachments[i].content`, NOT `m.content` — assistant-ui keeps
        // composer-typed content and attachment content as separate fields
        // on ThreadUserMessage (verified in the installed
        // @assistant-ui/core@0.1.17's runtime/utils/thread-message-like.ts:
        // `fromThreadMessageLike` builds `content` from the composer's typed
        // parts only and stores attachments separately with their own
        // `content` array). So the existing `p.type === "text"` filter over
        // `m.content` never sees attachment text — it must be pulled in
        // explicitly, prepended so the doc block precedes the user's
        // question (the injection block's trailing "\n\n" separates them).
        const apiMessages = messages.map((m) => {
          let attachmentText =
            m.role === "user"
              ? m.attachments
                  .flatMap((a) => a.content)
                  .filter((p) => p.type === "text")
                  .map((p) => (p as { type: "text"; text: string }).text)
                  .join("")
              : "";

          if (attachmentText.length > MAX_ATTACHMENT_CHARS) {
            const marker = "\n[attachments truncated]\n\n";
            attachmentText =
              attachmentText.slice(0, MAX_ATTACHMENT_CHARS - marker.length) +
              marker;
          }

          const bodyText = m.content
            .filter((p) => p.type === "text")
            .map((p) => (p as { type: "text"; text: string }).text)
            .join("");

          return { role: m.role, content: attachmentText + bodyText };
        });

        const res = await fetch("/api/chat", {
          method: "POST",
          signal: abortSignal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: apiMessages,
            modelId: selectedModel,
            systemPrompt: SYSTEM_PROMPTS[activePrompt],
            temperature,
            maxTokens,
          }),
        });

        if (!res.ok || !res.body) {
          throw new Error(
            `Chat request failed: ${res.status} ${res.statusText}`,
          );
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let reasoning = "";
        let content = "";

        const yieldParts = () => {
          const parts: { type: "reasoning" | "text"; text: string }[] = [];
          if (reasoning) parts.push({ type: "reasoning", text: reasoning });
          if (content) parts.push({ type: "text", text: content });
          return { content: parts };
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          // NDJSON: one event per line. Hold a partial line in `buf`.
          let nl = buf.indexOf("\n");
          while (nl >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (line) {
              try {
                const ev = JSON.parse(line) as {
                  type: "reasoning" | "content";
                  delta: string;
                };
                if (ev.type === "reasoning") reasoning += ev.delta;
                else content += ev.delta;
              } catch {
                // Tolerate the legacy plain-text stream by treating any
                // non-JSON line as a content delta.
                content += line;
              }
            }
            nl = buf.indexOf("\n");
          }
          yield yieldParts();
        }

        // Flush trailing partial line, if any.
        if (buf.trim()) {
          try {
            const ev = JSON.parse(buf.trim()) as {
              type: "reasoning" | "content";
              delta: string;
            };
            if (ev.type === "reasoning") reasoning += ev.delta;
            else content += ev.delta;
          } catch {
            content += buf;
          }
          yield yieldParts();
        }
      },
    }),
    [selectedModel, activePrompt, temperature, maxTokens],
  );

  const runtime = useLocalRuntime(adapter, {
    adapters: { attachments: documentAttachmentAdapter },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}
