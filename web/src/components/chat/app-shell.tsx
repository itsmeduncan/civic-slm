"use client";

import { useEffect, useState } from "react";
import { Thread } from "@/components/assistant-ui/thread";
import { ChatRuntimeProvider } from "./runtime-provider";
import { EmptyState } from "./empty-state";
import { Header } from "./header";
import { Inspector } from "./inspector";
import { Sidebar } from "./sidebar";
import { MODELS, type PromptKey } from "./types";
import { useHealth } from "./use-health";

export function AppShell() {
  const [selectedModel, setSelectedModel] = useState<string>(MODELS[0].id);
  const [activePrompt, setActivePrompt] = useState<PromptKey>("general");
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [threadKey, setThreadKey] = useState(0);
  const health = useHealth();

  // Cmd+/ cycles task presets
  useEffect(() => {
    const order: PromptKey[] = [
      "general",
      "extraction",
      "factcheck",
      "summarize",
    ];
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "/") {
        e.preventDefault();
        setActivePrompt((cur) => {
          const i = order.indexOf(cur);
          return order[(i + 1) % order.length];
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <ChatRuntimeProvider
      key={threadKey}
      selectedModel={selectedModel}
      activePrompt={activePrompt}
      temperature={temperature}
      maxTokens={maxTokens}
    >
      <div className="flex h-svh w-full bg-background text-foreground">
        <Sidebar
          activePrompt={activePrompt}
          onPromptChange={setActivePrompt}
          onNewChat={() => setThreadKey((k) => k + 1)}
          health={health}
        />

        <div className="flex min-w-0 flex-1 flex-col">
          <Header
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            activePrompt={activePrompt}
            temperature={temperature}
            onTemperatureChange={setTemperature}
            maxTokens={maxTokens}
            onMaxTokensChange={setMaxTokens}
            inspectorOpen={inspectorOpen}
            onToggleInspector={() => setInspectorOpen((v) => !v)}
          />

          <main className="relative flex flex-1 overflow-hidden">
            <div className="relative flex flex-1 flex-col overflow-hidden bg-gradient-to-b from-background to-muted/20">
              <Thread
                empty={
                  <EmptyState
                    activePrompt={activePrompt}
                    onPromptChange={setActivePrompt}
                  />
                }
              />
            </div>
            {inspectorOpen && (
              <Inspector
                selectedModel={selectedModel}
                activePrompt={activePrompt}
                temperature={temperature}
                maxTokens={maxTokens}
                health={health}
              />
            )}
          </main>
        </div>
      </div>
    </ChatRuntimeProvider>
  );
}
