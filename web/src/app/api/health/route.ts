const BASE_URL = process.env.CIVIC_SLM_LM_STUDIO_URL ?? "http://127.0.0.1:1234";

export async function GET() {
  const started = performance.now();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${BASE_URL}/v1/models`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) {
      return Response.json(
        {
          ok: false,
          status: res.status,
          baseUrl: BASE_URL,
          latencyMs: Math.round(performance.now() - started),
        },
        { status: 200 },
      );
    }
    const data = (await res.json()) as {
      data?: { id: string; object?: string }[];
    };
    return Response.json({
      ok: true,
      baseUrl: BASE_URL,
      models: (data.data ?? []).map((m) => m.id),
      latencyMs: Math.round(performance.now() - started),
    });
  } catch (err) {
    return Response.json(
      {
        ok: false,
        baseUrl: BASE_URL,
        error: err instanceof Error ? err.message : String(err),
        latencyMs: Math.round(performance.now() - started),
      },
      { status: 200 },
    );
  }
}
