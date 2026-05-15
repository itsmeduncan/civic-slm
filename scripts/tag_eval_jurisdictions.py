"""One-off script: tag every eval example with its target jurisdiction.

Closes the analysis half of #25 (second-city held-out eval). The current
eval JSONLs lack a `jurisdiction` field, so without this we can't slice
per-jurisdiction scores. Adding the field to `_EvalBase` and backfilling
350+ committed examples is a v0.3.x schema change; for the v1.1
post-hoc analysis we keep the data card immutable and stash the tags
in a sidecar at `data/eval/.jurisdiction-tags.jsonl` (gitignored —
regenerable, not corpus state).

Cost: ~350 Claude Sonnet 4.6 calls at ~$0.005 each, roughly $1.75 total.

Run once after the eval corpus changes. Idempotent — re-running skips
already-tagged ids.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from civic_slm.llm.backend import AnthropicBackend
from civic_slm.logging import configure, get_logger

log = get_logger(__name__)

PROMPT = """You are tagging civic-NLP eval examples with the U.S. jurisdiction
they target. Given the text below (a meeting agenda excerpt, staff report,
ordinance section, etc.), return EXACTLY one of these slugs:

  san-clemente, austin, houston, nyc, phoenix, seattle, cook-county,
  cuyahoga-county, atlanta, boston, denver, portland-or, generic

Use the most-specific match. "generic" means the text is jurisdiction-agnostic
(no state-specific statute names, no city-specific landmarks/streets, no
named-municipality fingerprints). Vocabulary hints:
- san-clemente: CEQA, CUP, El Camino Real, Avenida Presidio, Pier, T-Pier
- austin: SUP (vs CUP), TIRZ, Lady Bird Lake, home-rule Texas
- houston: Harris County
- nyc: ULURP, CEQR, community board, City Council file int.
- phoenix: City of Phoenix, Valley Metro
- seattle: King County, Sound Transit, HALA
- cook-county: Cook County board, Forest Preserve
- cuyahoga-county: Cuyahoga, Cleveland
- atlanta: Fulton County, BeltLine, Atlanta City Council
- boston: Suffolk County, BPDA, Back Bay, Fenway
- denver: RTD, 16th Street Mall, Denver City Council
- portland-or: TriMet, Metro, Portland (Oregon)

Return ONLY the slug. No prose, no quotes, no markdown.

---
Text:
{ctx}
"""

_LEGAL_SLUGS = {
    "san-clemente",
    "austin",
    "houston",
    "nyc",
    "phoenix",
    "seattle",
    "cook-county",
    "cuyahoga-county",
    "atlanta",
    "boston",
    "denver",
    "portland-or",
    "generic",
}


def _ctx_for(ex: dict[str, object], bench: str) -> str:
    if bench == "extraction":
        return str(ex.get("document_text", ""))
    parts: list[str] = []
    q = ex.get("question") or ex.get("prompt")
    if q:
        parts.append(f"Question: {q}")
    c = ex.get("context")
    if c:
        parts.append(f"Context: {c}")
    return "\n".join(parts)


async def tag_one(backend: AnthropicBackend, ex_id: str, ctx: str) -> str:
    text = await backend.complete(system=None, user=PROMPT.format(ctx=ctx[:6000]), max_tokens=20)
    slug = text.strip().lower().replace('"', "").replace("`", "")
    # Coerce LLM drift back to the legal set.
    if slug not in _LEGAL_SLUGS:
        for legal in _LEGAL_SLUGS:
            if legal in slug:
                slug = legal
                break
        else:
            log.warning("juris_tag_drift", ex_id=ex_id, raw=text[:100])
            slug = "generic"
    return slug


async def main() -> None:
    configure()
    out_path = Path("data/eval/.jurisdiction-tags.jsonl")
    already: dict[str, str] = {}
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    already[row["id"]] = row["jurisdiction"]
        log.info("resuming_tags", existing=len(already))

    backend = AnthropicBackend(model="claude-sonnet-4-6")
    sem = asyncio.Semaphore(4)

    work: list[tuple[str, str, str]] = []  # (id, bench, ctx)
    for fn, bench in [
        ("civic_factuality.jsonl", "factuality"),
        ("refusal.jsonl", "refusal"),
        ("structured_extraction.jsonl", "extraction"),
        ("side_by_side.jsonl", "side_by_side"),
    ]:
        p = Path("data/eval") / fn
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if ex["id"] in already:
                    continue
                work.append((ex["id"], bench, _ctx_for(ex, bench)))

    log.info("tagging_start", to_tag=len(work), already=len(already))

    async def one(ex_id: str, bench: str, ctx: str) -> tuple[str, str, str]:
        async with sem:
            try:
                slug = await tag_one(backend, ex_id, ctx)
            except Exception as exc:
                log.warning("tag_failed", ex_id=ex_id, error=str(exc)[:200])
                slug = "generic"
            return ex_id, bench, slug

    results = await asyncio.gather(*[one(i, b, c) for i, b, c in work])

    # Append-only write so re-runs preserve prior tags.
    with out_path.open("a", encoding="utf-8") as fh:
        for ex_id, bench, slug in results:
            fh.write(json.dumps({"id": ex_id, "bench": bench, "jurisdiction": slug}) + "\n")

    counts: dict[str, int] = {}
    for _, _, slug in results:
        counts[slug] = counts.get(slug, 0) + 1
    print(f"tagged {len(results)} new examples (+{len(already)} from resume):")
    for s, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s:18s} {n}")
    print(f"\nsidecar: {out_path}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
