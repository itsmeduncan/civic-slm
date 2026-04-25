"""San Clemente, CA — demo recipe.

Used as the reference recipe. Copy `_template.py` (not this file) to add a new
jurisdiction — that file has a cleaner commented skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass

from civic_slm.ingest.harness import DiscoveredDoc
from civic_slm.ingest.recipes._browser import run_browser_agent

INSTRUCTION = """\
Open https://www.san-clemente.org/ and navigate to the City Council meetings page.
Find the list of past City Council meetings going back to {since}, up to {max_docs} most recent.
For each meeting, return:
  - title: the meeting name (e.g. "City Council Meeting")
  - meeting_date: the date in YYYY-MM-DD format
  - source_url: the direct URL to the PDF agenda for that meeting
Skip workshops, special meetings, and closed sessions. Only return regular City Council agendas.
Return strictly as a JSON array of objects with the three keys above.
"""


@dataclass(frozen=True)
class SanClementeRecipe:
    jurisdiction: str = "san-clemente"
    state: str = "CA"
    instruction: str = INSTRUCTION

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        return await run_browser_agent(self.instruction, since=since, max_docs=max_docs)
