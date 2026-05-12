import { OpenAI } from "openai";

// Match the project-wide convention from docs/RUNTIMES.md.
const BASE_URL =
  process.env.CIVIC_SLM_CANDIDATE_URL ?? "http://127.0.0.1:8080/v1";
const API_KEY = process.env.CIVIC_SLM_LLM_API_KEY ?? "not-required";

// Map the UI's stable slugs to whatever string the local server expects.
// Override per-slot via env to match your loaded models without code edits.
const MODEL_MAP: Record<string, string> = {
  "gemma-4": process.env.CIVIC_SLM_GEMMA_MODEL ?? "gemma-4-31b-it-mlx",
  "qwen-civic": process.env.CIVIC_SLM_CIVIC_MODEL ?? "civic-slm-qwen2.5-7b",
  "qwen-base":
    process.env.CIVIC_SLM_CANDIDATE_MODEL ??
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
};

const openai = new OpenAI({ apiKey: API_KEY, baseURL: BASE_URL });

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export async function POST(req: Request) {
  const {
    messages,
    modelId,
    systemPrompt,
  }: { messages: ChatMessage[]; modelId?: string; systemPrompt?: string } =
    await req.json();

  const fullMessages: ChatMessage[] = systemPrompt
    ? [{ role: "system", content: systemPrompt }, ...messages]
    : messages;

  const resolvedModel =
    (modelId && MODEL_MAP[modelId]) || modelId || "local-model";

  const completion = await openai.chat.completions.create({
    model: resolvedModel,
    stream: true,
    messages: fullMessages,
  });

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of completion) {
          const delta = chunk.choices[0]?.delta?.content;
          if (delta) controller.enqueue(encoder.encode(delta));
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}
