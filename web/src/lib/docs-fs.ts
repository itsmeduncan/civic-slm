import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

export interface DocEntry {
  slug: string;
  title: string;
  description: string;
  file: string; // relative path from repo root
}

export const DOCS: DocEntry[] = [
  {
    slug: "readme",
    title: "Overview",
    description:
      "What civic-slm is, who it's for, and the high-level pipeline.",
    file: "README.md",
  },
  {
    slug: "usage",
    title: "End-to-end usage",
    description:
      "Crawl → synth → train → eval → ship, with copy-pasteable commands.",
    file: "docs/USAGE.md",
  },
  {
    slug: "recipes",
    title: "Adding a jurisdiction",
    description:
      "How to author a recipe for a new U.S. city, county, or township.",
    file: "docs/RECIPES.md",
  },
  {
    slug: "runtimes",
    title: "Runtimes",
    description:
      "Serving the candidate model via MLX, Ollama, LM Studio, or llama.cpp.",
    file: "docs/RUNTIMES.md",
  },
  {
    slug: "glossary",
    title: "Glossary",
    description: "Civic-domain and ML jargon, demystified.",
    file: "docs/GLOSSARY.md",
  },
  {
    slug: "sources",
    title: "Sources & ToS",
    description: "Per-jurisdiction crawl scope and terms-of-service notes.",
    file: "docs/SOURCES.md",
  },
];

const REPO_ROOT = path.resolve(process.cwd(), "..");

export function findDoc(slug: string): DocEntry | undefined {
  return DOCS.find((d) => d.slug === slug);
}

export async function readDoc(entry: DocEntry): Promise<string> {
  const abs = path.join(REPO_ROOT, entry.file);
  return readFile(abs, "utf8");
}
