import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { DocShell } from "@/components/docs/doc-shell";
import { DOCS } from "@/lib/docs-fs";

export const metadata = {
  title: "Docs · Civic SLM",
  description: "Documentation for civic-slm.",
};

export default function DocsIndexPage() {
  return (
    <DocShell>
      <header className="mb-8">
        <p className="text-xs uppercase tracking-wider text-muted-foreground">
          Documentation
        </p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">
          Everything in one place
        </h1>
        <p className="mt-2 max-w-xl text-sm text-muted-foreground">
          The same docs that live under{" "}
          <code className="rounded bg-muted px-1 font-mono text-xs">docs/</code>{" "}
          in the repo, rendered for easy reading while you experiment in the
          chat.
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-2">
        {DOCS.map((d) => (
          <Link
            key={d.slug}
            href={`/docs/${d.slug}`}
            className="group rounded-xl border border-border/60 bg-card p-4 transition-all hover:border-border hover:shadow-sm"
          >
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold tracking-tight">
                {d.title}
              </h2>
              <ArrowRight className="size-4 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {d.description}
            </p>
          </Link>
        ))}
      </div>
    </DocShell>
  );
}
