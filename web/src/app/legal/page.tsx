import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";
import { DocShell } from "@/components/docs/doc-shell";
import { Prose } from "@/components/docs/prose";

export const metadata = {
  title: "Terms & license · Civic SLM",
  description:
    "Use terms, privacy notes, model-output disclaimer, and the MIT license for civic-slm.",
};

async function readLicense(): Promise<string> {
  const abs = path.resolve(process.cwd(), "..", "LICENSE");
  try {
    return await readFile(abs, "utf8");
  } catch {
    return "MIT License — see the repository for the canonical text.";
  }
}

export default async function LegalPage() {
  const license = await readLicense();

  const body = `
## What this is

This web playground is a **local development tool** for dogfooding the civic-slm
candidate model. It is not a hosted service, has no user accounts, no
persistence, and no analytics. Everything you type stays on your machine and
travels to whichever local runtime you have configured
(LM Studio / llama.cpp / MLX).

## Acceptable use

By using this playground you agree to:

- Use civic documents you have a legal right to access (see
  [Sources & ToS](/docs/sources) for per-jurisdiction notes).
- Not use the model to produce or distribute material that is illegal, harmful,
  defamatory, or that misrepresents official government positions.
- Treat model output as **suggestion, not authority**. Do not rely on it for
  legal, financial, regulatory, or safety-critical decisions without a human
  expert in the loop.

## Model output disclaimer

Civic SLM is a small language model fine-tuned on public civic documents. It
can — and will — produce confidently wrong answers. Verify every claim against
the underlying source document before acting on it. The maintainers make no
warranty about accuracy, completeness, currency, or fitness for any particular
purpose. See the MIT license below for the full disclaimer.

## Privacy

- **No telemetry.** This UI does not phone home.
- **No accounts.** There is nothing to sign up for.
- **Local inference by default.** When \`CIVIC_SLM_LLM_BACKEND=local\` (the
  default), prompts and completions only ever leave the machine if you
  explicitly point the runtime at a remote endpoint.
- **Anthropic-backed runs.** If you set \`CIVIC_SLM_LLM_BACKEND=anthropic\`,
  prompts you send through synthetic-data or judge pipelines are transmitted to
  Anthropic's API under their terms. The chat playground itself does not call
  Anthropic.

## Trademarks & jurisdictions named

References to specific U.S. cities, counties, school districts, and platforms
(Granicus, Legistar, CivicPlus, Municode, etc.) are descriptive only. The
maintainers are not affiliated with, endorsed by, or sponsored by any
jurisdiction or vendor named in this project.

## Contributions

Pull requests are welcome. By contributing you agree to license your
contribution under the same MIT license below.

## License

The civic-slm source code is released under the MIT License:

\`\`\`
${license.trim()}
\`\`\`

The fine-tuned model weights and synthetic datasets, when released, will carry
their own licenses (declared in the corresponding model card and dataset card
on Hugging Face). Base-model and source-document licenses still apply.
`.trimStart();

  return (
    <DocShell variant="legal">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-wider text-muted-foreground">
          Legal
        </p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">
          Terms & license
        </h1>
        <p className="mt-2 max-w-xl text-sm text-muted-foreground">
          A short, plain-English read covering acceptable use, the model-output
          disclaimer, privacy, and the underlying MIT license.
        </p>
      </header>
      <Prose>{body}</Prose>
    </DocShell>
  );
}
