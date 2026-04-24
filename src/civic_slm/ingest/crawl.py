"""CLI entry: `python -m civic_slm.ingest.crawl --jurisdiction san-clemente --state CA`.

The `--jurisdiction` slug must be registered in `_RECIPES` below. Look up by
slug alone — recipes carry their own `state`, so we don't need a state arg
unless two jurisdictions share a slug across states (rare; we'd disambiguate
then).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.harness import Recipe, crawl
from civic_slm.ingest.recipes.san_clemente import SanClementeRecipe
from civic_slm.logging import configure, get_logger

app = typer.Typer(help="Crawl civic documents for a given U.S. jurisdiction.")
log = get_logger(__name__)

_RECIPES: dict[str, Callable[[], Recipe]] = {
    "san-clemente": SanClementeRecipe,
}


@app.command()
def main(
    jurisdiction: str = typer.Option(
        ..., "--jurisdiction", "--city", help="Jurisdiction slug, e.g. `san-clemente`."
    ),
    since: str = typer.Option("2025-01-01", help="ISO date — earliest meeting to include."),
    max_docs: int = typer.Option(20, "--max", help="Max docs to crawl this run."),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
) -> None:
    configure()
    if jurisdiction not in _RECIPES:
        raise typer.BadParameter(f"unknown jurisdiction {jurisdiction!r}; have: {sorted(_RECIPES)}")
    recipe = _RECIPES[jurisdiction]()
    target = data_dir or settings().data_dir
    landed = asyncio.run(crawl(recipe=recipe, data_dir=target, since=since, max_docs=max_docs))
    log.info(
        "crawl_complete",
        jurisdiction=recipe.jurisdiction,
        state=recipe.state,
        landed=len(landed),
    )


if __name__ == "__main__":
    app()
