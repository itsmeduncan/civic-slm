import { OpenAI } from "openai";

// Match the project-wide convention from docs/RUNTIMES.md. LM Studio's
// developer server listens on :1234 by default; we append /v1 here so the
// OpenAI client builds /v1/chat/completions correctly.
const RAW_BASE = process.env.CIVIC_SLM_CANDIDATE_URL ?? "http://127.0.0.1:1234";
const BASE_URL = RAW_BASE.endsWith("/v1") ? RAW_BASE : `${RAW_BASE}/v1`;
const API_KEY = process.env.CIVIC_SLM_LLM_API_KEY ?? "not-required";

// Map the UI's stable slugs to whatever string the local server expects.
// Override per-slot via env to match your loaded models without code edits.
const MODEL_MAP: Record<string, string> = {
  "gemma-4": process.env.CIVIC_SLM_GEMMA_MODEL ?? "gemma-4-31b-it-mlx",
  "qwen-civic": process.env.CIVIC_SLM_CIVIC_MODEL ?? "qwen3.6-27b-ud-mlx",
  "qwen-base": process.env.CIVIC_SLM_CANDIDATE_MODEL ?? "qwen3.6-27b-ud-mlx",
};

const openai = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL });

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

// Stream protocol: NDJSON, one event per line. The two event kinds are:
//   {"type":"reasoning","delta":"..."}   — chain-of-thought token (reasoning models)
//   {"type":"content","delta":"..."}     — visible response token
// Reasoning ones arrive first on Qwen 3.6 / Gemma 4; emitting them through to
// the client gives the user a "thinking…" indicator that isn't a placeholder.
export async function POST(req: Request) {
  const {
    messages,
    modelId,
    systemPrompt,
    temperature,
    maxTokens,
  }: {
    messages: ChatMessage[];
    modelId?: string;
    systemPrompt?: string;
    temperature?: number;
    maxTokens?: number;
  } = await req.json();

  const fullMessages: ChatMessage[] = systemPrompt
    ? [{ role: "system", content: systemPrompt }, ...messages]
    : messages;

  const resolvedModel =
    (modelId && MODEL_MAP[modelId]) || modelId || "local-model";

  const completion = await openai.chat.completions.create({
    model: resolvedModel,
    stream: true,
    messages: fullMessages,
    temperature: temperature ?? 0.2,
    max_tokens: maxTokens ?? 4096,
  });

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (type: "reasoning" | "content", delta: string) => {
        controller.enqueue(
          encoder.encode(JSON.stringify({ type, delta }) + "\n"),
        );
      };
      try {
        for await (const chunk of completion) {
          const delta = chunk.choices[0]?.delta as
            | { content?: string | null; reasoning_content?: string | null }
            | undefined;
          if (!delta) continue;
          if (delta.reasoning_content)
            emit("reasoning", delta.reasoning_content);
          if (delta.content) emit("content", delta.content);
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}
