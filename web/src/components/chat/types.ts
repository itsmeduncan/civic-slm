export type PromptKey = "general" | "extraction" | "factcheck" | "summarize";

export const SYSTEM_PROMPTS: Record<PromptKey, string> = {
  general:
    "You are a helpful civic assistant specialized in U.S. local government documents.",
  extraction:
    "You are a structured data extractor. Extract information from civic documents into clean, flat JSON. Output ONLY the JSON object, no prose.",
  factcheck:
    "You are a civic fact-checker. Every claim you make must be supported by a verbatim citation from the provided text. If the answer is not in the text, state that you cannot find it.",
  summarize:
    "You are a civic summarizer. Provide concise, bulleted summaries of municipal documents, highlighting fiscal impacts and key deadlines.",
};

export type TaskPreset = {
  key: PromptKey;
  label: string;
  tagline: string;
  description: string;
  starters: string[];
};

export const TASK_PRESETS: TaskPreset[] = [
  {
    key: "general",
    label: "General",
    tagline: "Open-ended Q&A grounded in civic context",
    description:
      "Ask anything about local-government documents. Best for orientation, definitions, and broad questions.",
    starters: [
      "What is a comprehensive plan and how does it differ from a zoning ordinance?",
      "Walk me through a typical city council agenda.",
      "Explain what CEQA exemption §15061(b)(3) means.",
    ],
  },
  {
    key: "extraction",
    label: "Extraction",
    tagline: "Documents → clean JSON, no prose",
    description:
      "Paste a staff report or agenda item. Get a flat JSON record back. No conversational filler.",
    starters: [
      "Extract the fiscal impact section as JSON.",
      "Pull out every motion + vote tally as a JSON array.",
      "Return {project_name, address, applicant, hearing_date} from this notice.",
    ],
  },
  {
    key: "factcheck",
    label: "Fact-check",
    tagline: "Every claim cited verbatim",
    description:
      "Give it context + a question. It will quote the exact passage or decline. No confabulation.",
    starters: [
      "Does this report say anything about water rationing? Cite the passage.",
      "What's the approved budget for parks maintenance? Verbatim citation only.",
      "Is the housing element compliance status mentioned? Quote it.",
    ],
  },
  {
    key: "summarize",
    label: "Summarize",
    tagline: "Bulleted, fiscal-first, deadline-aware",
    description:
      "Long documents → concise bullets. Surfaces fiscal impacts and key deadlines first.",
    starters: [
      "Summarize this 50-page staff report in 8 bullets, fiscal first.",
      "Give me the TL;DR of this agenda with deadlines flagged.",
      "Three-bullet summary of the public comment section.",
    ],
  },
];

export const PROMPT_META: Record<PromptKey, TaskPreset> = Object.fromEntries(
  TASK_PRESETS.map((t) => [t.key, t]),
) as Record<PromptKey, TaskPreset>;

export type ModelSlot = {
  id: string;
  name: string;
  family: "qwen" | "civic" | "gemma";
  blurb: string;
  badge?: string;
};

export const MODELS: ModelSlot[] = [
  {
    id: "qwen-base",
    name: "Qwen 3.6 27B",
    family: "qwen",
    blurb:
      "Project base model. Reasoning model — generous max-tokens recommended.",
    badge: "Base",
  },
  {
    id: "qwen-civic",
    name: "Civic SLM v1",
    family: "civic",
    blurb:
      "Fine-tuned on civic corpora. Cites verbatim, refuses when grounded.",
    badge: "Trained",
  },
  {
    id: "gemma-4",
    name: "Gemma 4 31B",
    family: "gemma",
    blurb: "Alternate base for side-by-side comparison.",
    badge: "Alt",
  },
];
