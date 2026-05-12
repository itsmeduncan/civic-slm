"use client";

import React, { useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { FileText, MessageSquare, ScanSearch, ShieldCheck } from "lucide-react";

const SYSTEM_PROMPTS = {
  general:
    "You are a helpful civic assistant specialized in U.S. local government documents.",
  extraction:
    "You are a structured data extractor. Your goal is to extract information from civic documents into clean, flat JSON. Do not include conversational filler.",
  factcheck:
    "You are a civic fact-checker. Every claim you make must be supported by a verbatim citation from the provided text. If the answer is not in the text, state that you cannot find it.",
  summarize:
    "You are a civic summarizer. Provide concise, bulleted summaries of municipal documents, highlighting fiscal impacts and key deadlines.",
} as const;

type PromptKey = keyof typeof SYSTEM_PROMPTS;

const PROMPT_META: Record<
  PromptKey,
  {
    label: string;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  general: {
    label: "General",
    description: "Helpful civic assistant.",
    icon: MessageSquare,
  },
  extraction: {
    label: "Extraction",
    description: "Flat JSON, no filler.",
    icon: ScanSearch,
  },
  factcheck: {
    label: "Fact-check",
    description: "Verbatim citations only.",
    icon: ShieldCheck,
  },
  summarize: {
    label: "Summarize",
    description: "Bulleted, with fiscal + deadlines.",
    icon: FileText,
  },
};

const MODELS = [
  { id: "gemma-4", name: "Gemma 4 (local)" },
  { id: "qwen-civic", name: "Civic SLM v1 (trained)" },
  { id: "qwen-base", name: "Qwen 2.5 (base)" },
] as const;

function ChatRuntime({
  selectedModel,
  activePrompt,
  children,
}: {
  selectedModel: string;
  activePrompt: PromptKey;
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
    [selectedModel, activePrompt],
  );

  const runtime = useLocalRuntime(adapter);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}

export default function ChatPage() {
  const [selectedModel, setSelectedModel] = useState<string>(MODELS[0].id);
  const [activePrompt, setActivePrompt] = useState<PromptKey>("general");

  return (
    <ChatRuntime selectedModel={selectedModel} activePrompt={activePrompt}>
      <div className="flex h-svh w-full flex-col bg-background text-foreground">
        <header className="flex h-14 shrink-0 items-center gap-4 border-b px-4">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-emerald-500" />
            <span className="font-semibold tracking-tight">Civic SLM</span>
            <span className="hidden text-xs text-muted-foreground sm:inline">
              local-government chat playground
            </span>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <Select
              value={selectedModel}
              onValueChange={(v) => v && setSelectedModel(v)}
            >
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent>
                {MODELS.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <aside className="hidden w-64 shrink-0 flex-col gap-1 border-r p-3 md:flex">
            <div className="px-2 pb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Task preset
            </div>
            {(Object.keys(PROMPT_META) as PromptKey[]).map((key) => {
              const meta = PROMPT_META[key];
              const Icon = meta.icon;
              const active = activePrompt === key;
              return (
                <Button
                  key={key}
                  variant={active ? "secondary" : "ghost"}
                  className={cn(
                    "h-auto w-full justify-start gap-3 px-3 py-2 text-left",
                    active && "ring-1 ring-border",
                  )}
                  onClick={() => setActivePrompt(key)}
                >
                  <Icon className="size-4 shrink-0" />
                  <span className="flex flex-col">
                    <span className="text-sm font-medium leading-tight">
                      {meta.label}
                    </span>
                    <span className="text-xs font-normal text-muted-foreground">
                      {meta.description}
                    </span>
                  </span>
                </Button>
              );
            })}

            <div className="mt-auto rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
              <div className="mb-1 font-medium text-foreground">
                Active system prompt
              </div>
              <p className="leading-snug">{SYSTEM_PROMPTS[activePrompt]}</p>
            </div>
          </aside>

          <main className="relative flex flex-1 flex-col overflow-hidden">
            <Thread />
          </main>
        </div>
      </div>
    </ChatRuntime>
  );
}
