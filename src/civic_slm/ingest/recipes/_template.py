"""Template recipe — copy this file to add a new U.S. jurisdiction.

Steps:
  1. Copy this file to `recipes/<your_jurisdiction>.py`.
  2. Rename the class (`MyJurisdictionRecipe`) and the module-level `INSTRUCTION`.
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
from civic_slm.schema import DocType

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
        """Run the browser-use agent. Identical to san_clemente.py — copy-paste safe.

        See `recipes/san_clemente.py` for the canonical implementation. The
        only thing you typically change is `INSTRUCTION` and the dataclass
        defaults; the discover body is shared boilerplate.
        """
        import os

        from browser_use import Agent  # type: ignore[import-not-found]

        prompt = self.instruction.format(since=since, max_docs=max_docs)
        backend_choice = os.environ.get("CIVIC_SLM_LLM_BACKEND", "anthropic").strip().lower()
        if backend_choice == "local":
            from browser_use.llm import ChatOpenAI  # type: ignore[import-not-found]

            llm = ChatOpenAI(  # pyright: ignore[reportUnknownVariableType]
                model=os.environ.get("CIVIC_SLM_LOCAL_LLM_MODEL", "default"),
                base_url=os.environ.get("CIVIC_SLM_LOCAL_LLM_URL", "http://127.0.0.1:8081") + "/v1",
                api_key="not-needed",
            )
        else:
            from browser_use.llm import ChatAnthropic  # type: ignore[import-not-found]

            from civic_slm.config import require

            llm = ChatAnthropic(  # pyright: ignore[reportUnknownVariableType]
                model="claude-sonnet-4-6",
                api_key=require("ANTHROPIC_API_KEY"),
            )

        agent = Agent(task=prompt, llm=llm)  # pyright: ignore[reportUnknownVariableType]
        result = await agent.run()  # pyright: ignore[reportUnknownMemberType]
        return _parse_result(result)


def _parse_result(result: object) -> list[DiscoveredDoc]:
    """Identical to san_clemente._parse_result; you usually don't touch this."""
    import json

    text = str(result)
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        items = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return []

    out: list[DiscoveredDoc] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("source_url")
        title = item.get("title")
        if not isinstance(url, str) or not isinstance(title, str):
            continue
        date = item.get("meeting_date")
        out.append(
            DiscoveredDoc(
                title=title,
                source_url=url,
                doc_type=DocType.AGENDA,
                meeting_date=date if isinstance(date, str) else None,
            )
        )
    return out
