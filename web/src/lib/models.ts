// MUST mirror src/civic_slm/serve/models.py — the project-side model registry.
// If you add a model on the Python side, add it here too. The web playground
// runs in a separate Node process, so it can't import the Python registry; the
// 4-row TypeScript mirror is acceptable duplication.
//
// `label` is the project-side identity (matches Python). `servedName` is what
// LM Studio publishes in /v1/models. They cannot disagree at a row.

export interface Model {
  label: string;
  servedName: string;
}

export const MODELS: Record<string, Model> = {
  "base-qwen3.6-27b": {
    label: "base-qwen3.6-27b",
    servedName: "qwen3.6-27b-ud-mlx",
  },
  "base-qwen2.5-7b": {
    label: "base-qwen2.5-7b",
    servedName: "qwen2.5-7b-instruct-mlx",
  },
  "comparator-gemma-4-31b": {
    label: "comparator-gemma-4-31b",
    servedName: "gemma-4-31b-it-mlx",
  },
  "base-gemma-4-31b": {
    label: "base-gemma-4-31b",
    servedName: "gemma-4-31b-it-mlx",
  },
  "civic-slm-v1": {
    label: "civic-slm-v1",
    servedName: "qwen3.6-27b-ud-mlx",
  },
} satisfies Record<string, Model>;

// UI dropdown slots → registry labels. Stable slugs for the playground.
export const SLOT_TO_LABEL: Record<string, string> = {
  "gemma-4": "comparator-gemma-4-31b",
  "qwen-civic": "civic-slm-v1",
  "qwen-base": "base-qwen3.6-27b",
} satisfies Record<string, string>;

export function resolve(label: string): Model {
  // Fallback: unregistered labels resolve to themselves on both sides — same
  // contract as the Python registry. No silent label/served divergence.
  return MODELS[label] ?? { label, servedName: label };
}
