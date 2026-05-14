"""YAML-driven recipe loader.

Why YAML: adding a U.S. jurisdiction is a data problem, not a code problem.
For ~80% of cities (CivicPlus, IQM2, Granicus / Legistar, PrimeGov) the
only differences between recipes are the slug, the state, the start URL,
and which vendor template to use as the agent's discovery prompt. A 10-line
YAML expresses all of that without anyone having to learn Pydantic or
async-Python.

A YAML recipe still satisfies the `Recipe` Protocol from `harness.py`,
so the rest of the pipeline (`harness.crawl`, `manifest.append`,
`civic-slm process`, `civic-slm synth`, training) is unchanged. The
recipe lookup in `crawl.py` auto-discovers `*.yaml` alongside the
existing Python recipes — no `_RECIPES` dict edit.

When YAML isn't enough (a vendor needs custom HTTP, a non-browser API
path, a custom JSON shape) the maintainer can copy `_template.py` and
write a Python recipe. The two paths coexist; YAML is the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from civic_slm.ingest.harness import DiscoveredDoc, DiscoveredVideo
from civic_slm.ingest.recipes._browser import run_browser_agent
from civic_slm.ingest.recipes._youtube import youtube_channel_videos
from civic_slm.schema import DocType

_VENDORS_DIR = Path(__file__).parent / "_vendors"

# Project-side label → template filename. Keep the mapping explicit so a
# typo in the YAML's `vendor:` field surfaces here rather than as a vague
# "template not found" deep inside browser-use.
_VENDOR_TEMPLATES: dict[str, str] = {
    "civicplus": "civicplus.md",
    "iqm2": "iqm2.md",
    "granicus": "granicus_legistar.md",
    "legistar": "granicus_legistar.md",
    "primegov": "primegov_municode.md",
    "municode": "primegov_municode.md",
}


class RecipeError(ValueError):
    """Raised when a YAML recipe is missing required fields or names an unknown vendor."""


@dataclass(frozen=True)
class YamlRecipe:
    """A jurisdiction recipe defined in YAML rather than Python.

    Satisfies the `Recipe` Protocol via duck-typing (frozen dataclass with
    `jurisdiction` / `state` attributes + `discover` / `discover_videos`).

    `instruction` is the fully-formatted browser-use prompt. The class
    method `from_yaml()` resolves the vendor template + start URL into
    this field at load time so `discover()` has no I/O beyond the call
    to `run_browser_agent()`.
    """

    jurisdiction: str
    state: str
    start_url: str
    instruction: str
    doc_type_default: DocType = DocType.AGENDA
    youtube_channel: str | None = None
    # Source path is kept for diagnostics (`civic-slm doctor` can print it);
    # not part of the Recipe protocol contract.
    source_path: Path | None = field(default=None, compare=False)

    @classmethod
    def from_yaml(cls, path: Path) -> YamlRecipe:
        """Parse a YAML recipe file and resolve the vendor template.

        Required keys: `jurisdiction`, `state`, `start_url`, `vendor`.
        Optional: `doc_type_default` (default `agenda`), `youtube_channel`,
        `instruction` (overrides the vendor template entirely).
        """
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        missing = [k for k in ("jurisdiction", "state", "start_url") if not data.get(k)]
        if missing:
            raise RecipeError(
                f"{path}: missing required field(s) {missing}. "
                "Required: jurisdiction, state, start_url, vendor."
            )

        instruction = data.get("instruction")
        if not instruction:
            vendor = (data.get("vendor") or "").strip().lower()
            if vendor not in _VENDOR_TEMPLATES:
                known = ", ".join(sorted(set(_VENDOR_TEMPLATES)))
                raise RecipeError(
                    f"{path}: vendor {vendor!r} is not recognized. "
                    f"Known vendors: {known}. Or provide an explicit `instruction:` field."
                )
            tpl_path = _VENDORS_DIR / _VENDOR_TEMPLATES[vendor]
            instruction = tpl_path.read_text(encoding="utf-8")

        # Resolve the start_url placeholder eagerly so `discover()` is a
        # one-line passthrough. The {since}/{max_docs} placeholders are
        # filled per-call inside `run_browser_agent`.
        instruction = instruction.replace("{start_url}", str(data["start_url"]))

        doc_type_default_raw = (data.get("doc_type_default") or "agenda").strip().lower()
        try:
            doc_type_default = DocType(doc_type_default_raw)
        except ValueError as exc:
            valid = ", ".join(sorted(t.value for t in DocType))
            raise RecipeError(
                f"{path}: doc_type_default {doc_type_default_raw!r} is not a known DocType. "
                f"Valid: {valid}."
            ) from exc

        return cls(
            jurisdiction=str(data["jurisdiction"]).strip(),
            state=str(data["state"]).strip().upper(),
            start_url=str(data["start_url"]).strip(),
            instruction=instruction,
            doc_type_default=doc_type_default,
            youtube_channel=(data.get("youtube_channel") or None),
            source_path=path,
        )

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        return await run_browser_agent(
            self.instruction,
            since=since,
            max_docs=max_docs,
            default_doc_type=self.doc_type_default,
        )

    async def discover_videos(self, *, since: str, max_videos: int) -> list[DiscoveredVideo]:
        """Return videos when `youtube_channel:` is set in the YAML; otherwise empty.

        The harness's `crawl_videos` treats an empty list as "this recipe doesn't
        do videos" — same as a Python recipe omitting the method.
        """
        if not self.youtube_channel:
            return []
        return youtube_channel_videos(self.youtube_channel, since=since, max_videos=max_videos)
