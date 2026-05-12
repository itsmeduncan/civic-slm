"use client";

import { useEffect, useState } from "react";

export type Health =
  | { state: "loading" }
  | {
      state: "ok";
      baseUrl: string;
      models: string[];
      latencyMs: number;
    }
  | {
      state: "down";
      baseUrl: string;
      error?: string;
      latencyMs: number;
    };

export function useHealth(intervalMs = 15_000): Health {
  const [health, setHealth] = useState<Health>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const data = (await res.json()) as {
          ok: boolean;
          baseUrl: string;
          models?: string[];
          error?: string;
          latencyMs: number;
        };
        if (cancelled) return;
        if (data.ok) {
          setHealth({
            state: "ok",
            baseUrl: data.baseUrl,
            models: data.models ?? [],
            latencyMs: data.latencyMs,
          });
        } else {
          setHealth({
            state: "down",
            baseUrl: data.baseUrl,
            error: data.error,
            latencyMs: data.latencyMs,
          });
        }
      } catch (err) {
        if (cancelled) return;
        setHealth({
          state: "down",
          baseUrl: "(unreachable)",
          error: err instanceof Error ? err.message : String(err),
          latencyMs: 0,
        });
      }
    };
    void ping();
    const t = setInterval(ping, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [intervalMs]);

  return health;
}
