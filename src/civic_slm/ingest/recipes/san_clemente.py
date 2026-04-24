"""San Clemente recipe: City Council agendas via browser-use.

San Clemente runs on Granicus today, but the recipe is intentionally
platform-agnostic — the browser-use agent reads the natural-language instruction
and figures out the navigation. If the city migrates to a different vendor,
this recipe should keep working without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from civic_slm.ingest.harness import DiscoveredDoc
from civic_slm.schema import DocType

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
    city: str = "san-clemente"
    instruction: str = INSTRUCTION

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        """Run the browser-use agent against the city site.

        Lazily imports browser-use so tests and module-level imports don't pay
        the cost (and don't require Chromium to be installed). Picks the
        agent's reasoning LLM based on `CIVIC_SLM_LLM_BACKEND`:
          - `anthropic` (default): ChatAnthropic on Claude Sonnet 4.6.
          - `local`: ChatOpenAI pointed at `CIVIC_SLM_LOCAL_LLM_URL`.
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
    """Coerce the agent's structured output into DiscoveredDoc list.

    browser-use returns a result envelope; we look for the final JSON array on
    the agent's output. Be liberal — wrap anything we can't parse in an empty
    list and log upstream.
    """
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
