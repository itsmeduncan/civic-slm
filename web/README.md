This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Document attachments

Attach a `.pdf`, `.txt`, or `.md` file in the chat composer and ask questions about it. Text is extracted server-side and injected as context for the model.

**Limits:**

- Max 10 MB per file
- Attach one or a few documents per turn (no hard file-count limit is enforced)
- Extracted text capped at 30,000 characters per file at the endpoint, **plus a combined 30,000-character total cap across all attachments on a turn** — truncated with a marker if exceeded

**Setup:** Attachments call the **RAG shim** — start it with `civic-slm rag serve` (defaults to `http://127.0.0.1:8767`). Override the URL the web app uses with the `CIVIC_SLM_RAG_URL` env var. Chat without attachments needs only LM Studio.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
