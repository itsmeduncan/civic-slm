"use client";

import { useMemo } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import type { PromptKey } from "./types";
import { SYSTEM_PROMPTS } from "./types";

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
        const apiMessages = messages.map((m) => ({
          role: m.role,
          content: m.content
            .filter((p) => p.type === "text")
            .map((p) => (p as { type: "text"; text: string }).text)
            .join(""),
        }));

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

  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}
