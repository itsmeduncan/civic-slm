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
        let text = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          text += decoder.decode(value, { stream: true });
          yield { content: [{ type: "text", text }] };
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
