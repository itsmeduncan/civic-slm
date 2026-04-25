"""Synthetic SFT generator.

For each input chunk x task, prompt Claude Opus 4.7 to emit N (system, input, output)
triples. Every emitted line is parsed and validated against `InstructionExample`
before it lands in `data/sft/v0.jsonl`. Validation failures are dropped with a
log line, not crashed on — synthesis is bulky and one bad chunk shouldn't tank
the whole run.

Why store the prompt SHA in provenance: lets us re-run only the examples that
came from a stale prompt template when we iterate on the prompts. The hash is
the cheapest possible cache key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from civic_slm.llm.backend import Backend, select_backend
from civic_slm.logging import get_logger
from civic_slm.schema import DocumentChunk, InstructionExample, Provenance, TaskType

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_MODEL = "claude-opus-4-7"

_TASK_TO_TEMPLATE: dict[TaskType, str] = {
    TaskType.QA_GROUNDED: "qa_grounded.md",
    TaskType.REFUSAL: "refusal.md",
    TaskType.EXTRACT: "extract.md",
    TaskType.SUMMARIZE: "summarize.md",
}


@dataclass(frozen=True)
class _Prompt:
    template: str
    sha: str

    @classmethod
    def load(cls, task: TaskType) -> _Prompt:
        path = PROMPTS_DIR / _TASK_TO_TEMPLATE[task]
        text = path.read_text(encoding="utf-8")
        return cls(template=text, sha=hashlib.sha256(text.encode("utf-8")).hexdigest())


async def generate_for_chunk(
    *,
    chunk: DocumentChunk,
    jurisdiction: str,
    state: str,
    doc_type: str,
    task: TaskType,
    n: int,
    backend: Backend | None = None,
) -> list[InstructionExample]:
    """Generate N InstructionExamples for one chunk + task. Drops invalid lines.

    Backend defaults to whatever `select_backend()` resolves from the
    `CIVIC_SLM_LLM_BACKEND` env var (anthropic or local). Pass an explicit
    backend in tests.
    """
    backend = backend or select_backend(default_anthropic_model=DEFAULT_MODEL)

    prompt = _Prompt.load(task)
    user = prompt.template.format(
        jurisdiction=jurisdiction,
        state=state,
        doc_type=doc_type,
        section_path=" > ".join(chunk.section_path) or "(none)",
        chunk_text=chunk.text,
        n=n,
    )

    text = await backend.complete(system=None, user=user, max_tokens=4096)
    generator = "claude" if "claude" in backend.model.lower() else "model_v0"
    return parse_examples(
        text=text,
        task=task,
        chunk_id=f"{chunk.doc_id}#{chunk.chunk_idx}",
        provenance=Provenance(
            generator=generator,
            model=backend.model,
            prompt_sha=prompt.sha,
            created_at=datetime.now(UTC),
        ),
    )


def parse_examples(
    *,
    text: str,
    task: TaskType,
    chunk_id: str,
    provenance: Provenance,
) -> list[InstructionExample]:
    """Parse model output into InstructionExamples; drop invalid lines.

    Output format from the prompts is one JSON object per line. We accept either
    line-by-line JSON, or a single fenced block — strip code fences then iterate.
    """
    cleaned = re.sub(r"```(json)?", "", text).strip()
    out: list[InstructionExample] = []
    for data in _iter_json_objects(cleaned):
        if not isinstance(data, dict):
            continue
        # Normalize output to a string — extract templates emit a JSON value.
        if isinstance(data.get("output"), (dict, list)):
            data["output"] = json.dumps(data["output"])
        merged = {
            "id": str(uuid.uuid4()),
            "task": task.value,
            "source_chunk_ids": [chunk_id],
            "provenance": provenance.model_dump(mode="json"),
            **data,
        }
        try:
            out.append(InstructionExample.model_validate(merged))
        except ValidationError as exc:
            log.warning("synth_drop_invalid", task=task.value, error=str(exc)[:200])
            continue
    return out


def _iter_json_objects(text: str) -> Iterable[object]:
    """Yield JSON objects from `text`. Accepts one-per-line *or* pretty-printed.

    We prefer one-per-line (the prompts ask for it), but models drift to
    multi-line indented JSON. Streaming `json.JSONDecoder.raw_decode` across
    the full string tolerates both without needing to pre-split by line.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        # Skip whitespace and stray commas.
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text):
            break
        # Only attempt to decode where a JSON value could start.
        if text[idx] not in '{["0123456789-tfn':
            idx += 1
            continue
        try:
            obj, offset = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        yield obj
        idx = offset


def write_jsonl(path: Path, examples: Iterable[InstructionExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
            n += 1
    return n


async def generate_corpus(
    *,
    chunks: list[DocumentChunk],
    jurisdiction: str,
    state: str,
    doc_type: str,
    out_path: Path,
    n_per_chunk: int = 3,
    tasks: tuple[TaskType, ...] = (
        TaskType.QA_GROUNDED,
        TaskType.REFUSAL,
        TaskType.EXTRACT,
        TaskType.SUMMARIZE,
    ),
    concurrency: int = 4,
    backend: Backend | None = None,
) -> int:
    """Generate examples for many chunks x tasks. Returns total examples written."""
    backend = backend or select_backend(default_anthropic_model=DEFAULT_MODEL)
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: DocumentChunk, task: TaskType) -> list[InstructionExample]:
        async with sem:
            try:
                return await generate_for_chunk(
                    chunk=chunk,
                    jurisdiction=jurisdiction,
                    state=state,
                    doc_type=doc_type,
                    task=task,
                    n=n_per_chunk,
                    backend=backend,
                )
            except Exception as exc:
                log.warning("synth_failed", chunk=chunk.doc_id, task=task.value, error=str(exc))
                return []

    tasks_coros = [one(c, t) for c in chunks for t in tasks]
    batches = await asyncio.gather(*tasks_coros)
    total = 0
    for batch in batches:
        total += write_jsonl(out_path, batch)
    log.info("synth_complete", total=total, out=str(out_path))
    return total
