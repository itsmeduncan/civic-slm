const RAW = process.env.CIVIC_SLM_RAG_URL ?? "http://127.0.0.1:8767";
const SHIM = RAW.replace(/\/$/, "");

export async function POST(req: Request): Promise<Response> {
  const form = await req.formData();
  const file = form.get("file");
  if (!(file instanceof File)) {
    return Response.json({ error: "no file field in upload" }, { status: 400 });
  }
  const forward = new FormData();
  forward.set("file", file, file.name);
  try {
    const res = await fetch(`${SHIM}/v1/attachments`, {
      method: "POST",
      body: forward,
    });
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json(
      { error: "RAG shim not running — start it with `civic-slm rag serve`" },
      { status: 503 },
    );
  }
}
