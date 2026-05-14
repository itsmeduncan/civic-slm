"""Tests for the YAML-driven recipe path and the crawl auto-discovery.

What's under test: dropping a YAML file into `recipes/` should produce a
discoverable, functional Recipe — no Python edit, no manual registration.
The browser-use call is mocked; we only assert that `YamlRecipe.discover()`
forwards a fully-formatted instruction to `run_browser_agent`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from civic_slm.ingest.crawl import _load_recipes
from civic_slm.ingest.recipes.yaml_recipe import (
    _VENDOR_TEMPLATES,
    RecipeError,
    YamlRecipe,
)
from civic_slm.schema import DocType

_RECIPES_DIR = Path("src/civic_slm/ingest/recipes")


def _write(tmp_path: Path, content: str, name: str = "tmpville.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_yaml_recipe_loads_with_vendor_template(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
jurisdiction: tmpville
state: ca
vendor: iqm2
start_url: https://tmpville.iqm2.com/Citizens/Calendar.aspx
""".lstrip(),
    )
    recipe = YamlRecipe.from_yaml(path)
    assert recipe.jurisdiction == "tmpville"
    # State is normalized to uppercase regardless of YAML case.
    assert recipe.state == "CA"
    assert recipe.doc_type_default == DocType.AGENDA
    # start_url placeholder is resolved at load time.
    assert "tmpville.iqm2.com" in recipe.instruction
    assert "{start_url}" not in recipe.instruction


def test_yaml_recipe_explicit_instruction_overrides_vendor(tmp_path: Path) -> None:
    """An explicit `instruction:` field replaces the vendor template entirely."""
    path = _write(
        tmp_path,
        """
jurisdiction: tmpville
state: CA
vendor: civicplus
start_url: https://tmpville.gov/AgendaCenter
instruction: |
  Do the thing. {since} {max_docs}
""".lstrip(),
    )
    recipe = YamlRecipe.from_yaml(path)
    # Explicit instruction wins; vendor template is not loaded.
    assert "Do the thing." in recipe.instruction
    assert "CivicPlus" not in recipe.instruction


def test_yaml_recipe_unknown_vendor_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
jurisdiction: tmpville
state: CA
vendor: not-a-real-vendor
start_url: https://example.org
""".lstrip(),
    )
    with pytest.raises(RecipeError, match="not recognized"):
        YamlRecipe.from_yaml(path)


def test_yaml_recipe_missing_required_field_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
state: CA
vendor: iqm2
start_url: https://example.org
""".lstrip(),
    )
    with pytest.raises(RecipeError, match="missing required field"):
        YamlRecipe.from_yaml(path)


def test_yaml_recipe_doc_type_validates(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
jurisdiction: tmpville
state: CA
vendor: iqm2
start_url: https://example.org
doc_type_default: not-a-doc-type
""".lstrip(),
    )
    with pytest.raises(RecipeError, match="DocType"):
        YamlRecipe.from_yaml(path)


@pytest.mark.parametrize("vendor", sorted(set(_VENDOR_TEMPLATES)))
def test_every_vendor_template_resolves_placeholders(vendor: str, tmp_path: Path) -> None:
    """No template should ship with an unresolved {start_url} after load."""
    path = _write(
        tmp_path,
        f"""
jurisdiction: tmpville
state: CA
vendor: {vendor}
start_url: https://example.org/calendar
""".lstrip(),
    )
    recipe = YamlRecipe.from_yaml(path)
    assert "{start_url}" not in recipe.instruction
    assert "https://example.org/calendar" in recipe.instruction


def test_yaml_recipe_discover_forwards_to_browser_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`discover()` should call `run_browser_agent` with the resolved instruction."""
    captured: dict[str, object] = {}

    async def fake_run_browser_agent(
        instruction: str,
        *,
        since: str,
        max_docs: int,
        default_doc_type: DocType,
    ) -> list[object]:
        captured["instruction"] = instruction
        captured["since"] = since
        captured["max_docs"] = max_docs
        captured["default_doc_type"] = default_doc_type
        return []

    monkeypatch.setattr(
        "civic_slm.ingest.recipes.yaml_recipe.run_browser_agent",
        fake_run_browser_agent,
    )
    path = _write(
        tmp_path,
        """
jurisdiction: tmpville
state: CA
vendor: civicplus
start_url: https://tmpville.gov/AgendaCenter
""".lstrip(),
    )
    recipe = YamlRecipe.from_yaml(path)
    asyncio.run(recipe.discover(since="2025-09-01", max_docs=3))
    assert "tmpville.gov/AgendaCenter" in captured["instruction"]
    assert captured["since"] == "2025-09-01"
    assert captured["max_docs"] == 3
    assert captured["default_doc_type"] == DocType.AGENDA


def test_load_recipes_finds_both_yaml_and_py() -> None:
    """The shipped recipes dir should yield at least san-clemente (.py) and santa-monica (.yaml)."""
    _load_recipes.cache_clear()  # type: ignore[attr-defined]
    recipes = _load_recipes()
    assert "san-clemente" in recipes  # Python recipe
    assert "santa-monica" in recipes  # YAML recipe
    sc = recipes["san-clemente"]()
    sm = recipes["santa-monica"]()
    assert type(sm).__name__ == "YamlRecipe"
    assert sc.state == "CA"
    assert sm.state == "CA"


def test_santa_monica_recipe_uses_iqm2_template() -> None:
    """End-to-end smoke against the shipped santa-monica.yaml."""
    path = _RECIPES_DIR / "santa-monica.yaml"
    assert path.exists(), "santa-monica.yaml must ship in the recipes dir"
    recipe = YamlRecipe.from_yaml(path)
    assert recipe.jurisdiction == "santa-monica"
    assert recipe.state == "CA"
    assert "santamonicacityca.iqm2.com" in recipe.instruction
    # IQM2 template-specific phrase that confirms the right template loaded.
    assert "IQM2" in recipe.instruction or "Calendar.aspx" in recipe.instruction
