"""`civic-slm data-card` — auto-generated per-jurisdiction corpus breakdown.

Closes #26. The training corpus crosses jurisdictions, vendors, and
doc-types; DATA_CARD.md has to track that without anyone manually
counting rows. This module scans the manifest and processed JSONLs,
groups by jurisdiction, and emits a markdown table — same shape every
time so it can drop into DATA_CARD.md between sentinel markers.

The output is intentionally read-only against `data/`: this is a
*report*, not a regeneration. Re-running it cannot mutate corpus state.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.manifest import load_manifest
from civic_slm.logging import configure
from civic_slm.schema import CivicDocument


@dataclass(frozen=True)
class JurisdictionStats:
    """Aggregates for one jurisdiction. Built from the manifest + processed/."""

    slug: str
    state: str
    doc_count: int
    chunk_count: int
    token_count: int
    doc_types: Counter[str]
    earliest: datetime | None
    latest: datetime | None

    @property
    def date_range(self) -> str:
        if self.earliest is None or self.latest is None:
            return "—"
        if self.earliest.date() == self.latest.date():
            return self.earliest.date().isoformat()
        return f"{self.earliest.date().isoformat()} → {self.latest.date().isoformat()}"

    @property
    def top_doc_types(self) -> str:
        """`agenda·30, minutes·5` — the dominant types, descending."""
        if not self.doc_types:
            return "—"
        items = self.doc_types.most_common()
        return ", ".join(f"{name}·{n}" for name, n in items)


def _chunk_stats_for(jurisdiction: str, data_dir: Path) -> tuple[int, int]:
    """Return `(chunk_count, token_count)` from `data/processed/{slug}.jsonl`.

    Reads via raw JSON rather than `load_chunks()` because that helper
    constructs the full Pydantic object — overkill when we only need two
    integer fields and want the report to stay fast on a large corpus.
    """
    path = data_dir / "processed" / f"{jurisdiction}.jsonl"
    if not path.exists():
        return 0, 0
    chunks = 0
    tokens = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # Skip malformed lines rather than fail the whole report
                # — the dump is informational, and a sister CI check
                # validates the JSONLs as a separate gate.
                continue
            chunks += 1
            tokens += int(obj.get("token_count", 0))
    return chunks, tokens


def compute_stats(data_dir: Path) -> list[JurisdictionStats]:
    """Group manifest entries by jurisdiction and attach chunk/token counts.

    Sorted by (state, slug) so two runs against the same data produce
    byte-identical markdown — important for `--check` mode under CI.
    """
    manifest = load_manifest(data_dir)
    by_juris: dict[str, list[CivicDocument]] = {}
    for doc in manifest:
        by_juris.setdefault(doc.jurisdiction, []).append(doc)

    out: list[JurisdictionStats] = []
    for slug, docs in by_juris.items():
        chunks, tokens = _chunk_stats_for(slug, data_dir)
        types: Counter[str] = Counter()
        for d in docs:
            types[d.doc_type.value] += 1
        retrieved = [d.retrieved_at for d in docs]
        out.append(
            JurisdictionStats(
                slug=slug,
                state=docs[0].state,
                doc_count=len(docs),
                chunk_count=chunks,
                token_count=tokens,
                doc_types=types,
                earliest=min(retrieved) if retrieved else None,
                latest=max(retrieved) if retrieved else None,
            )
        )
    return sorted(out, key=lambda s: (s.state, s.slug))


def render_markdown(stats: list[JurisdictionStats]) -> str:
    """Render the per-jurisdiction breakdown as a markdown table.

    The output is wrapped in HTML sentinel comments so a future `--write`
    flag can splice it into DATA_CARD.md without clobbering the human-
    written prose around it.
    """
    lines = [
        "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->",
        "",
        "| Slug | State | Docs | Chunks | Tokens | Doc types | Crawl range |",
        "| ---- | ----- | ---- | ------ | ------ | --------- | ----------- |",
    ]
    if not stats:
        lines.append("| _(none — manifest is empty)_ |   |   |   |   |   |   |")
    else:
        for s in stats:
            lines.append(
                f"| `{s.slug}` | {s.state} | {s.doc_count} | {s.chunk_count} | "
                f"{s.token_count:,} | {s.top_doc_types} | {s.date_range} |"
            )

    totals = _totals(stats)
    lines.extend(
        [
            f"| **total** | — | **{totals['docs']}** | **{totals['chunks']}** | "
            f"**{totals['tokens']:,}** | — | — |",
            "",
            "<!-- DATA_CARD:JURISDICTIONS:END -->",
        ]
    )
    return "\n".join(lines) + "\n"


def _totals(stats: list[JurisdictionStats]) -> dict[str, int]:
    return {
        "docs": sum(s.doc_count for s in stats),
        "chunks": sum(s.chunk_count for s in stats),
        "tokens": sum(s.token_count for s in stats),
    }


_BEGIN_MARKER = "<!-- DATA_CARD:JURISDICTIONS:BEGIN -->"
_END_MARKER = "<!-- DATA_CARD:JURISDICTIONS:END -->"


def _splice(target: Path, fresh_block: str) -> str:
    """Splice `fresh_block` between the sentinels in `target`.

    Raises if the sentinels aren't both present so a typo in DATA_CARD.md
    fails loud at write time instead of silently appending duplicates.
    """
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Add a `<!-- DATA_CARD:JURISDICTIONS:BEGIN -->` /"
            f" `END` block to it first, then re-run with --write."
        )
    text = target.read_text(encoding="utf-8")
    if _BEGIN_MARKER not in text or _END_MARKER not in text:
        raise ValueError(
            f"{target} is missing the sentinel block. Add lines containing "
            f"{_BEGIN_MARKER!r} and {_END_MARKER!r} where the table should land."
        )
    pre, rest = text.split(_BEGIN_MARKER, 1)
    _, post = rest.split(_END_MARKER, 1)
    # Idempotent splice: normalize the surrounding blank lines so a second
    # --write doesn't keep adding a newline. Exactly one blank line on each
    # side of the auto-generated block.
    pre_normalized = pre.rstrip("\n") + "\n\n"
    post_normalized = "\n\n" + post.lstrip("\n")
    return pre_normalized + fresh_block.strip("\n") + post_normalized


def main(
    write: bool = typer.Option(
        False,
        "--write",
        help=(
            "Splice the rendered table into DATA_CARD.md between the "
            "`<!-- DATA_CARD:JURISDICTIONS:BEGIN -->` / `END` sentinels. "
            "Without this flag, the markdown is printed to stdout."
        ),
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=(
            "Exit non-zero if DATA_CARD.md's table would change. Useful "
            "as a CI gate: a corpus change that's not reflected in the "
            "data card fails the build."
        ),
    ),
    target: Path = typer.Option(
        Path("DATA_CARD.md"),
        "--target",
        help="Markdown file to splice into (or check against).",
    ),
    data_dir: Path | None = typer.Option(
        None, "--data-dir", help="Override data dir (default: <repo>/data)."
    ),
) -> None:
    """Emit an auto-generated per-jurisdiction corpus breakdown.

    Reads `data/raw/manifest.jsonl` plus `data/processed/{slug}.jsonl`,
    groups by jurisdiction, and prints a markdown table. Pass `--write`
    to splice it into `DATA_CARD.md`; pass `--check` for the CI gate.
    """
    configure()
    target_dir = data_dir or settings().data_dir
    stats = compute_stats(target_dir)
    block = render_markdown(stats)

    if check:
        if not target.exists():
            typer.echo(f"{target} not found", err=True)
            raise typer.Exit(code=1)
        try:
            expected = _splice(target, block)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        actual = target.read_text(encoding="utf-8")
        if expected != actual:
            typer.echo(
                f"{target} is out of date. Run `civic-slm data-card --write` "
                f"and commit the result.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"{target} is up to date.")
        return

    if write:
        new_text = _splice(target, block)
        target.write_text(new_text, encoding="utf-8")
        typer.echo(f"Wrote {target}")
        return

    typer.echo(block, nl=False)


if __name__ == "__main__":
    typer.run(main)
