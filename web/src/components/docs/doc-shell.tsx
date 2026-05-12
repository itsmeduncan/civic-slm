import Link from "next/link";
import { ArrowLeft, BookOpen, Scale, Sparkles } from "lucide-react";
import { DOCS } from "@/lib/docs-fs";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

interface DocShellProps {
  activeSlug?: string;
  variant?: "docs" | "legal";
  children: React.ReactNode;
}

export function DocShell({
  activeSlug,
  variant = "docs",
  children,
}: DocShellProps) {
  return (
    <div className="flex h-svh w-full bg-background text-foreground">
      <aside className="hidden h-full w-64 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground md:flex">
        <Link
          href="/"
          className="flex h-14 items-center gap-2 border-b px-4 transition-colors hover:bg-sidebar-accent"
        >
          <Sparkles className="size-4" />
          <span className="font-semibold tracking-tight">Civic SLM</span>
        </Link>

        <div className="px-3 py-3">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="size-3.5" /> back to chat
          </Link>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pb-3">
          <SectionLabel>
            <BookOpen className="size-3" /> Docs
          </SectionLabel>
          <div className="mt-1 flex flex-col gap-0.5">
            {DOCS.map((d) => (
              <DocLink
                key={d.slug}
                href={`/docs/${d.slug}`}
                title={d.title}
                active={variant === "docs" && activeSlug === d.slug}
              />
            ))}
          </div>

          <SectionLabel className="mt-6">
            <Scale className="size-3" /> Legal
          </SectionLabel>
          <div className="mt-1 flex flex-col gap-0.5">
            <DocLink
              href="/legal"
              title="Terms & license"
              active={variant === "legal"}
            />
          </div>
        </nav>

        <Separator />
        <div className="px-3 py-3 text-[11px] text-muted-foreground">
          Documentation is served straight from the repo. Edits in{" "}
          <code className="rounded bg-muted px-1 font-mono">docs/</code> show up
          here.
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-6 py-10 sm:px-10">
          {children}
        </div>
      </main>
    </div>
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
        "flex items-center gap-1.5 px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  );
}

function DocLink({
  href,
  title,
  active,
}: {
  href: string;
  title: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "rounded-md px-2 py-1.5 text-sm transition-colors",
        "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
        active &&
          "bg-sidebar-accent text-sidebar-accent-foreground ring-1 ring-sidebar-border",
      )}
    >
      {title}
    </Link>
  );
}
