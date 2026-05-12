import { notFound } from "next/navigation";
import { DocShell } from "@/components/docs/doc-shell";
import { Prose } from "@/components/docs/prose";
import { DOCS, findDoc, readDoc } from "@/lib/docs-fs";

export function generateStaticParams() {
  return DOCS.map((d) => ({ slug: d.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const doc = findDoc(slug);
  if (!doc) return {};
  return {
    title: `${doc.title} · Civic SLM`,
    description: doc.description,
  };
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const doc = findDoc(slug);
  if (!doc) notFound();

  let body: string;
  try {
    body = await readDoc(doc);
  } catch {
    notFound();
  }

  return (
    <DocShell activeSlug={slug}>
      <header className="mb-2">
        <p className="text-xs uppercase tracking-wider text-muted-foreground">
          {doc.file}
        </p>
      </header>
      <Prose>{body}</Prose>
    </DocShell>
  );
}
