"""CLI entry: `python -m civic_slm.ingest.crawl --city san-clemente --max 20`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.harness import Recipe, crawl
from civic_slm.ingest.recipes.san_clemente import SanClementeRecipe
from civic_slm.logging import configure, get_logger

app = typer.Typer(help="Crawl civic documents for a given city.")
log = get_logger(__name__)

_RECIPES: dict[str, Callable[[], Recipe]] = {
    "san-clemente": SanClementeRecipe,
}


@app.command()
def main(
    city: str = typer.Option(..., help="City slug, e.g. `san-clemente`."),
    since: str = typer.Option("2025-01-01", help="ISO date — earliest meeting to include."),
    max_docs: int = typer.Option(20, "--max", help="Max docs to crawl this run."),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
) -> None:
    configure()
    if city not in _RECIPES:
        raise typer.BadParameter(f"unknown city {city!r}; have: {sorted(_RECIPES)}")
    recipe = _RECIPES[city]()
    target = data_dir or settings().data_dir
    landed = asyncio.run(crawl(recipe=recipe, data_dir=target, since=since, max_docs=max_docs))
    log.info("crawl_complete", city=city, landed=len(landed))


if __name__ == "__main__":
    app()
