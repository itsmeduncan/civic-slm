"""Template recipe — copy this file to add a new U.S. jurisdiction.

Steps:
  1. Copy this file to `recipes/<your_jurisdiction>.py`.
  2. Rename the class (`MyJurisdictionRecipe`).
  3. Set `jurisdiction` to a kebab-case slug and `state` to the 2-letter postal code.
  4. Edit `INSTRUCTION` — describe in plain English where to navigate and what
     to return. The browser-use agent reads this and figures out the rest.
  5. Register your class in `src/civic_slm/ingest/crawl.py`'s `_RECIPES` dict.

A good instruction:
  - Names the start URL.
  - Describes what to look for in user-visible terms ("council meeting agendas",
    not "DOM nodes matching `.agenda-link`").
  - States exclusions clearly ("skip workshops, special meetings, closed sessions").
  - Specifies the output JSON shape: `[{"title": ..., "meeting_date": "YYYY-MM-DD",
    "source_url": "https://..."}]`.
  - Stays under ~250 tokens. Long instructions confuse browser-use.
"""

from __future__ import annotations

from dataclasses import dataclass

from civic_slm.ingest.harness import DiscoveredDoc
from civic_slm.ingest.recipes._browser import run_browser_agent

INSTRUCTION = """\
Open https://example-city.gov/ and navigate to the City Council meetings page.
Find the list of past City Council meetings going back to {since}, up to {max_docs}
most recent. For each meeting, return:
  - title: the meeting name
  - meeting_date: the date in YYYY-MM-DD format
  - source_url: the direct URL to the PDF agenda
Skip workshops, special meetings, and closed sessions.
Return strictly as a JSON array of objects with the three keys above.
"""


@dataclass(frozen=True)
class TemplateRecipe:
    """Replace this docstring with a one-line description of the jurisdiction."""

    jurisdiction: str = "example-city"
    state: str = "XX"  # 2-letter U.S. postal code: CA, TX, NY, ...
    instruction: str = INSTRUCTION

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        return await run_browser_agent(self.instruction, since=since, max_docs=max_docs)
