"use client";

import { useState } from "react";
import {
  Check,
  ChevronDown,
  PanelRightClose,
  PanelRightOpen,
  Settings2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { ModelSlot, PromptKey } from "./types";
import { MODELS, PROMPT_META } from "./types";

export function Header({
  selectedModel,
  onSelectModel,
  activePrompt,
  temperature,
  onTemperatureChange,
  maxTokens,
  onMaxTokensChange,
  inspectorOpen,
  onToggleInspector,
}: {
  selectedModel: string;
  onSelectModel: (id: string) => void;
  activePrompt: PromptKey;
  temperature: number;
  onTemperatureChange: (value: number) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
}) {
  const model = MODELS.find((m) => m.id === selectedModel) ?? MODELS[0];
  const preset = PROMPT_META[activePrompt];

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background px-4">
      <div className="flex min-w-0 items-center gap-2">
        <h1 className="truncate text-sm font-semibold tracking-tight">
          {preset.label}
        </h1>
        <span className="hidden truncate text-xs text-muted-foreground sm:inline">
          · {preset.tagline}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <ModelPicker selected={model} onSelect={onSelectModel} />
        <ParamsPopover
          temperature={temperature}
          onTemperatureChange={onTemperatureChange}
          maxTokens={maxTokens}
          onMaxTokensChange={onMaxTokensChange}
        />
        <Separator orientation="vertical" className="mx-1 h-6" />
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={onToggleInspector}
          aria-label="Toggle inspector"
        >
          {inspectorOpen ? (
            <PanelRightClose className="size-4" />
          ) : (
            <PanelRightOpen className="size-4" />
          )}
        </Button>
      </div>
    </header>
  );
}

function ModelPicker({
  selected,
  onSelect,
}: {
  selected: ModelSlot;
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <Button variant="outline" size="sm" className="h-8 gap-1.5 px-2.5">
            <span className="font-mono text-xs">{selected.name}</span>
            {selected.badge && (
              <Badge
                variant="secondary"
                className="h-4 px-1 text-[10px] font-medium"
              >
                {selected.badge}
              </Badge>
            )}
            <ChevronDown className="size-3.5 text-muted-foreground" />
          </Button>
        }
      />
      <PopoverContent align="end" className="w-[320px] p-1">
        <div className="px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Model
        </div>
        {MODELS.map((m) => {
          const active = m.id === selected.id;
          return (
            <button
              key={m.id}
              onClick={() => {
                onSelect(m.id);
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-start gap-3 rounded-md px-2 py-2 text-left transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
              )}
            >
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="flex items-center gap-2">
                  <span className="font-mono text-sm">{m.name}</span>
                  {m.badge && (
                    <Badge variant="secondary" className="h-4 px-1 text-[10px]">
                      {m.badge}
                    </Badge>
                  )}
                </span>
                <span className="mt-0.5 text-xs text-muted-foreground">
                  {m.blurb}
                </span>
              </div>
              <Check
                className={cn(
                  "mt-1 size-4 shrink-0",
                  active ? "opacity-100" : "opacity-0",
                )}
              />
            </button>
          );
        })}
      </PopoverContent>
    </Popover>
  );
}

function ParamsPopover({
  temperature,
  onTemperatureChange,
  maxTokens,
  onMaxTokensChange,
}: {
  temperature: number;
  onTemperatureChange: (value: number) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
}) {
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            aria-label="Sampling parameters"
          >
            <Settings2 className="size-4" />
          </Button>
        }
      />
      <PopoverContent align="end" className="w-[320px]">
        <div className="space-y-4">
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <label className="text-xs font-medium">Temperature</label>
              <span className="font-mono text-xs text-muted-foreground">
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => onTemperatureChange(Number(e.target.value))}
              className="w-full accent-foreground"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              0 = deterministic, 1+ = creative. Extraction & fact-check work
              best near 0.
            </p>
          </div>

          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <label className="text-xs font-medium">Max tokens</label>
              <span className="font-mono text-xs text-muted-foreground">
                {maxTokens}
              </span>
            </div>
            <input
              type="range"
              min={256}
              max={8192}
              step={256}
              value={maxTokens}
              onChange={(e) => onMaxTokensChange(Number(e.target.value))}
              className="w-full accent-foreground"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Reasoning models (Qwen 3.6, Gemma 4) need generous budgets — their
              hidden thinking eats the first chunk.
            </p>
          </div>

          <div className="rounded-md bg-muted/40 p-2 text-[10px] text-muted-foreground">
            <span className="font-medium text-foreground">Shortcut · </span>
            <Kbd>⌘</Kbd> <Kbd>K</Kbd> opens the command bar (coming soon).
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
