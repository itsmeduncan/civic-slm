"use client";

import { Activity, BookOpen, Gauge, ScrollText, Wand2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { PromptKey } from "./types";
import { MODELS, PROMPT_META, SYSTEM_PROMPTS } from "./types";
import type { Health } from "./use-health";

export function Inspector({
  selectedModel,
  activePrompt,
  temperature,
  maxTokens,
  health,
}: {
  selectedModel: string;
  activePrompt: PromptKey;
  temperature: number;
  maxTokens: number;
  health: Health;
}) {
  const model = MODELS.find((m) => m.id === selectedModel) ?? MODELS[0];
  const preset = PROMPT_META[activePrompt];

  return (
    <aside className="hidden h-full w-[320px] shrink-0 flex-col border-l bg-background lg:flex">
      <Tabs defaultValue="run" className="flex h-full flex-col gap-0">
        <TabsList className="h-12 w-full justify-start rounded-none border-b bg-transparent px-2">
          <TabsTrigger
            value="run"
            className="gap-1.5 data-[state=active]:bg-muted"
          >
            <Activity className="size-3.5" /> Run
          </TabsTrigger>
          <TabsTrigger
            value="prompt"
            className="gap-1.5 data-[state=active]:bg-muted"
          >
            <Wand2 className="size-3.5" /> Prompt
          </TabsTrigger>
          <TabsTrigger
            value="docs"
            className="gap-1.5 data-[state=active]:bg-muted"
          >
            <BookOpen className="size-3.5" /> Docs
          </TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1">
          <TabsContent value="run" className="m-0 px-4 py-4">
            <Section title="Model">
              <Row
                label="Slot"
                value={
                  <>
                    <span className="font-mono">{model.name}</span>
                    {model.badge && (
                      <Badge
                        variant="secondary"
                        className="ml-1.5 h-4 px-1 text-[10px]"
                      >
                        {model.badge}
                      </Badge>
                    )}
                  </>
                }
              />
              <Row
                label="Family"
                value={<span className="font-mono">{model.family}</span>}
              />
              <p className="mt-2 text-xs text-muted-foreground">
                {model.blurb}
              </p>
            </Section>

            <Separator className="my-4" />

            <Section title="Sampling">
              <Row
                label="Temperature"
                value={
                  <span className="font-mono">{temperature.toFixed(2)}</span>
                }
              />
              <Row
                label="Max tokens"
                value={<span className="font-mono">{maxTokens}</span>}
              />
            </Section>

            <Separator className="my-4" />

            <Section
              title="Runtime"
              icon={<Gauge className="size-3 text-muted-foreground" />}
            >
              <Row
                label="Status"
                value={
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className={
                        health.state === "ok"
                          ? "size-1.5 rounded-full bg-emerald-500"
                          : health.state === "loading"
                            ? "size-1.5 animate-pulse rounded-full bg-zinc-400"
                            : "size-1.5 rounded-full bg-rose-500"
                      }
                    />
                    <span className="font-mono text-xs">
                      {health.state === "ok"
                        ? "online"
                        : health.state === "loading"
                          ? "checking"
                          : "offline"}
                    </span>
                  </span>
                }
              />
              <Row
                label="Base URL"
                value={
                  <span className="font-mono text-[11px] text-muted-foreground">
                    {health.state === "loading" ? "—" : health.baseUrl}
                  </span>
                }
              />
              {health.state === "ok" && (
                <Row
                  label="Latency"
                  value={
                    <span className="font-mono">{health.latencyMs}ms</span>
                  }
                />
              )}
              {health.state === "ok" && health.models.length > 0 && (
                <div className="mt-2 space-y-1">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    loaded
                  </div>
                  {health.models.map((m) => (
                    <div
                      key={m}
                      className="rounded border bg-muted/30 px-1.5 py-0.5 font-mono text-[11px]"
                    >
                      {m}
                    </div>
                  ))}
                </div>
              )}
              {health.state === "down" && health.error && (
                <p className="mt-2 rounded border border-rose-500/30 bg-rose-500/5 p-2 text-[11px] text-rose-700 dark:text-rose-300">
                  {health.error}
                </p>
              )}
            </Section>
          </TabsContent>

          <TabsContent value="prompt" className="m-0 px-4 py-4">
            <Section
              title={`${preset.label} preset`}
              icon={<ScrollText className="size-3 text-muted-foreground" />}
            >
              <p className="text-xs text-muted-foreground">
                {preset.description}
              </p>
              <pre className="mt-3 max-h-[260px] overflow-auto whitespace-pre-wrap rounded-md border bg-muted/30 p-3 font-mono text-[11px] leading-relaxed">
                {SYSTEM_PROMPTS[activePrompt]}
              </pre>
            </Section>

            <Separator className="my-4" />

            <Section title="Starters">
              <div className="space-y-1.5">
                {preset.starters.map((s) => (
                  <div
                    key={s}
                    className="rounded-md border bg-card px-2 py-1.5 text-xs text-muted-foreground"
                  >
                    {s}
                  </div>
                ))}
              </div>
            </Section>
          </TabsContent>

          <TabsContent value="docs" className="m-0 space-y-3 px-4 py-4 text-sm">
            <p className="text-muted-foreground">
              civic-slm is a domain fine-tune of Qwen 3.6 27B for U.S.
              local-government documents. The playground talks to LM Studio on{" "}
              <span className="font-mono text-foreground">:1234</span>; no
              third-party tokens are sent.
            </p>
            <DocsLink
              href="https://github.com/itsmeduncan/civic-slm/blob/main/docs/USAGE.md"
              label="USAGE.md — end-to-end walkthrough"
            />
            <DocsLink
              href="https://github.com/itsmeduncan/civic-slm/blob/main/docs/RUNTIMES.md"
              label="RUNTIMES.md — env, strict-local, side-by-side"
            />
            <DocsLink
              href="https://github.com/itsmeduncan/civic-slm/blob/main/MODEL_CARD.md"
              label="MODEL_CARD.md — baselines + intended use"
            />
            <DocsLink
              href="https://github.com/itsmeduncan/civic-slm/blob/main/docs/RECIPES.md"
              label="RECIPES.md — add a new jurisdiction"
            />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </aside>
  );
}

function Section({
  title,
  children,
  icon,
}: {
  title: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {icon}
        {title}
      </div>
      {children}
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right">{value}</span>
    </div>
  );
}

function DocsLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="flex items-center justify-between rounded-md border bg-card px-2.5 py-2 text-xs hover:bg-accent hover:text-accent-foreground"
    >
      <span className="font-mono">{label}</span>
    </a>
  );
}
