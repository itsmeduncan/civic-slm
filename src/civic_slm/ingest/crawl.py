"""CLI entry: `python -m civic_slm.ingest.crawl --jurisdiction san-clemente --state CA`.

The `--jurisdiction` slug is resolved by scanning the `recipes/` directory:

  * `recipes/*.yaml` — declarative recipes (the default, see `YamlRecipe`)
  * `recipes/*.py` (except `_*.py`) — custom Python recipes when YAML
    isn't expressive enough

No `_RECIPES` dict to maintain. Drop a YAML in `recipes/`, the next
`civic-slm crawl <slug>` finds it. Look up by slug alone — recipes
carry their own `state`, so we don't need a state arg unless two
jurisdictions share a slug across states (rare; we'd disambiguate then).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import typer

from civic_slm.config import settings
from civic_slm.ingest.harness import Recipe, crawl, crawl_videos
from civic_slm.ingest.recipes.yaml_recipe import YamlRecipe
from civic_slm.logging import configure, get_logger

app = typer.Typer(help="Crawl civic documents for a given U.S. jurisdiction.")
log = get_logger(__name__)

_RECIPES_DIR = Path(__file__).parent / "recipes"


@lru_cache(maxsize=1)
def _load_recipes() -> dict[str, Callable[[], Recipe]]:
    """Scan `recipes/` and build the slug → factory map.

    YAML recipes win when a slug collides with a Python recipe (the YAML
    path is the supported maintainer surface; a colliding `.py` is
    almost always a stale debug file). The collision is logged so it's
    discoverable in `civic-slm doctor` output.

    Python recipes are loaded by importing the module and walking its
    classes for ones that satisfy the `Recipe` protocol (have an async
    `discover` and a `jurisdiction` attribute). One class per file is the
    convention.
    """
    out: dict[str, Callable[[], Recipe]] = {}

    # Python recipes first; YAML overrides on collision.
    for py_path in sorted(_RECIPES_DIR.glob("*.py")):
        if py_path.name.startswith("_"):
            continue
        module_name = f"civic_slm.ingest.recipes.{py_path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            # One broken recipe shouldn't disable `civic-slm crawl --list`
            # for every other jurisdiction. Log and skip; the maintainer
            # sees the offender in the warning and can fix it in isolation.
            log.warning(
                "recipe_import_failed",
                module=module_name,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module_name:
                continue
            if not hasattr(obj, "jurisdiction") or not hasattr(obj, "discover"):
                continue
            try:
                instance = obj()
            except TypeError:
                # Recipe classes can require args; skip — those aren't auto-discoverable.
                continue
            slug = getattr(instance, "jurisdiction", None)
            if isinstance(slug, str) and slug:
                out[slug] = obj  # bind the class as the factory

    for yaml_path in sorted(_RECIPES_DIR.glob("*.yaml")):
        recipe = YamlRecipe.from_yaml(yaml_path)
        slug = recipe.jurisdiction
        if slug in out:
            log.info("recipe_yaml_overrides_py", slug=slug, yaml=str(yaml_path))
        out[slug] = lambda r=recipe: r  # already an instance; close over it

    return out


@app.command()
def main(
    jurisdiction: str = typer.Argument(..., help="Jurisdiction slug, e.g. `san-clemente`."),
    since: str = typer.Option("2025-01-01", help="ISO date — earliest meeting to include."),
    max_docs: int = typer.Option(20, "--max", help="Max docs to crawl this run."),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
) -> None:
    configure()
    recipes = _load_recipes()
    if jurisdiction not in recipes:
        raise typer.BadParameter(f"unknown jurisdiction {jurisdiction!r}; have: {sorted(recipes)}")
    recipe = recipes[jurisdiction]()
    target = data_dir or settings().data_dir
    landed = asyncio.run(crawl(recipe=recipe, data_dir=target, since=since, max_docs=max_docs))
    log.info(
        "crawl_complete",
        jurisdiction=recipe.jurisdiction,
        state=recipe.state,
        landed=len(landed),
    )


def videos_main(
    jurisdiction: str = typer.Argument(..., help="Jurisdiction slug, e.g. `san-clemente`."),
    since: str = typer.Option("2025-01-01", help="ISO date — earliest video upload to include."),
    max_videos: int = typer.Option(20, "--max", help="Max videos to crawl this run."),
    data_dir: Path | None = typer.Option(None, help="Override data dir."),
) -> None:
    """Discover meeting videos, fetch audio + captions, transcribe, append to manifest."""
    configure()
    recipes = _load_recipes()
    if jurisdiction not in recipes:
        raise typer.BadParameter(f"unknown jurisdiction {jurisdiction!r}; have: {sorted(recipes)}")
    recipe = recipes[jurisdiction]()
    target = data_dir or settings().data_dir
    landed = asyncio.run(
        crawl_videos(recipe=recipe, data_dir=target, since=since, max_videos=max_videos)
    )
    log.info(
        "crawl_videos_complete",
        jurisdiction=recipe.jurisdiction,
        state=recipe.state,
        landed=len(landed),
    )


if __name__ == "__main__":
    app()
