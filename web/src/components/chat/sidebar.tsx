"use client";

import {
  BookOpen,
  Code2,
  FileText,
  MessageSquare,
  Plus,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { PromptKey } from "./types";
import { TASK_PRESETS } from "./types";
import type { Health } from "./use-health";

const TASK_ICONS: Record<PromptKey, LucideIcon> = {
  general: MessageSquare,
  extraction: ScanSearch,
  factcheck: ShieldCheck,
  summarize: FileText,
};

export function Sidebar({
  activePrompt,
  onPromptChange,
  onNewChat,
  health,
}: {
  activePrompt: PromptKey;
  onPromptChange: (key: PromptKey) => void;
  onNewChat: () => void;
  health: Health;
}) {
  return (
    <aside className="hidden h-full w-64 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground md:flex">
      <div className="flex h-14 items-center gap-2 border-b px-4">
        <Sparkles className="size-4" />
        <span className="font-semibold tracking-tight">Civic SLM</span>
        <Badge variant="outline" className="ml-auto h-5 px-1.5 text-[10px]">
          local
        </Badge>
      </div>

      <div className="px-3 py-3">
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start gap-2"
          onClick={onNewChat}
        >
          <Plus className="size-4" />
          New chat
        </Button>
      </div>

      <ScrollArea className="flex-1 px-3">
        <SectionLabel>Task preset</SectionLabel>
        <div className="mt-1 flex flex-col gap-0.5">
          {TASK_PRESETS.map((preset) => {
            const Icon = TASK_ICONS[preset.key];
            const active = preset.key === activePrompt;
            return (
              <button
                key={preset.key}
                onClick={() => onPromptChange(preset.key)}
                className={cn(
                  "group flex items-start gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors",
                  "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                  active &&
                    "bg-sidebar-accent text-sidebar-accent-foreground ring-1 ring-sidebar-border",
                )}
              >
                <Icon
                  className={cn(
                    "mt-0.5 size-4 shrink-0",
                    active ? "text-foreground" : "text-muted-foreground",
                  )}
                />
                <span className="flex min-w-0 flex-col">
                  <span className="text-sm font-medium leading-tight">
                    {preset.label}
                  </span>
                  <span className="line-clamp-1 text-xs text-muted-foreground">
                    {preset.tagline}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        <SectionLabel className="mt-5">Recent</SectionLabel>
        <p className="mt-1 rounded-md border border-dashed px-2 py-3 text-xs text-muted-foreground">
          Thread history isn&apos;t persisted yet. This panel will list past
          conversations once the persistence layer lands.
        </p>
      </ScrollArea>

      <Separator />

      <div className="space-y-3 px-3 py-3 text-xs">
        <HealthRow health={health} />
        <div className="flex items-center gap-3 text-muted-foreground">
          <a
            href="https://github.com/itsmeduncan/civic-slm"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground"
          >
            <Code2 className="size-3.5" /> Repo
          </a>
          <a
            href="https://github.com/itsmeduncan/civic-slm/blob/main/docs/USAGE.md"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground"
          >
            <BookOpen className="size-3.5" /> Docs
          </a>
        </div>
      </div>
    </aside>
  );
}

function SectionLabel({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  );
}

function HealthRow({ health }: { health: Health }) {
  const dotClass =
    health.state === "ok"
      ? "bg-emerald-500"
      : health.state === "loading"
        ? "bg-zinc-400 animate-pulse"
        : "bg-rose-500";
  const label =
    health.state === "ok"
      ? `LM Studio · ${health.latencyMs}ms`
      : health.state === "loading"
        ? "checking…"
        : "LM Studio offline";
  const sub =
    health.state === "ok"
      ? `${health.models.length} model${health.models.length === 1 ? "" : "s"} loaded`
      : health.state === "down"
        ? health.baseUrl
        : "";

  return (
    <div className="flex items-center gap-2 text-muted-foreground">
      <span className={cn("size-2 shrink-0 rounded-full", dotClass)} />
      <span className="flex min-w-0 flex-col leading-tight">
        <span className="font-mono text-[11px] text-foreground">{label}</span>
        {sub && <span className="line-clamp-1 text-[10px]">{sub}</span>}
      </span>
    </div>
  );
}
