"use client";

import { useComposerRuntime } from "@assistant-ui/react";
import { ArrowUpRight, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Kbd } from "@/components/ui/kbd";
import { cn } from "@/lib/utils";
import type { PromptKey } from "./types";
import { PROMPT_META, TASK_PRESETS } from "./types";

export function EmptyState({
  activePrompt,
  onPromptChange,
}: {
  activePrompt: PromptKey;
  onPromptChange: (key: PromptKey) => void;
}) {
  const composer = useComposerRuntime();
  const active = PROMPT_META[activePrompt];

  const send = (text: string) => {
    composer.setText(text);
    composer.send();
  };

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-6 py-12">
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 rounded-full border bg-background px-2.5 py-1 text-xs text-muted-foreground">
          <Sparkles className="size-3.5" />
          civic-slm <span className="text-foreground/40">/</span>{" "}
          <span className="font-mono">{active.label.toLowerCase()}</span>
        </div>
        <h2 className="text-3xl font-semibold tracking-tight">
          {active.tagline}
        </h2>
        <p className="max-w-xl text-sm text-muted-foreground">
          {active.description}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {TASK_PRESETS.map((preset) => {
          const isActive = preset.key === activePrompt;
          return (
            <Card
              key={preset.key}
              className={cn(
                "cursor-pointer border-border/60 transition-all hover:border-border hover:shadow-sm",
                isActive && "border-foreground/30 bg-muted/30 shadow-sm",
              )}
              onClick={() => onPromptChange(preset.key)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-sm font-semibold">
                    {preset.label}
                  </CardTitle>
                  {isActive && (
                    <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                      active
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  {preset.tagline}
                </p>
              </CardHeader>
              <CardContent className="space-y-1 pt-0">
                {preset.starters.slice(0, 2).map((s) => (
                  <button
                    key={s}
                    onClick={(e) => {
                      e.stopPropagation();
                      onPromptChange(preset.key);
                      send(s);
                    }}
                    className={cn(
                      "group/starter flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5",
                      "text-left text-xs text-muted-foreground transition-colors",
                      "hover:bg-accent hover:text-accent-foreground",
                    )}
                  >
                    <span className="line-clamp-2">{s}</span>
                    <ArrowUpRight className="size-3.5 shrink-0 opacity-0 transition-opacity group-hover/starter:opacity-100" />
                  </button>
                ))}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <Kbd>↵</Kbd> send
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Kbd>⇧</Kbd> + <Kbd>↵</Kbd> newline
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Kbd>⌘</Kbd> + <Kbd>/</Kbd> task preset
        </span>
        <span className="ml-auto inline-flex items-center gap-1.5">
          served locally by LM Studio · zero tokens to a third party
        </span>
      </div>

      <Button
        variant="ghost"
        size="sm"
        className="self-start text-xs text-muted-foreground hover:text-foreground"
        onClick={() => send(active.starters[0])}
      >
        Try a starter <ArrowUpRight className="ml-1 size-3.5" />
      </Button>
    </div>
  );
}
