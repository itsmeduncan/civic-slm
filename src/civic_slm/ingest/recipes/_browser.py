"""Shared browser-use agent runner for every recipe.

Every jurisdiction recipe needs the same three things: (1) select the agent's
reasoning LLM based on `CIVIC_SLM_LLM_BACKEND`, (2) kick off the agent with the
formatted instruction, (3) parse its final JSON array into `DiscoveredDoc`s.
We lift all three here so a recipe file is ~15 lines instead of 60.

Adding a new jurisdiction means: write an `INSTRUCTION` string, set the
`jurisdiction`/`state` slug, and call `run_browser_agent(self.instruction, ...)`
from `discover`. That's it.
"""

from __future__ import annotations

import json
import os

from civic_slm.ingest.harness import DiscoveredDoc
from civic_slm.schema import DocType


async def run_browser_agent(
    instruction_template: str,
    *,
    since: str,
    max_docs: int,
    default_doc_type: DocType = DocType.AGENDA,
) -> list[DiscoveredDoc]:
    """Drive a `browser_use.Agent` against a site and parse its JSON result.

    Caller provides an `INSTRUCTION` string with `{since}` and `{max_docs}`
    placeholders plus guidance for the agent to return a JSON array of
    `{title, meeting_date, source_url}` objects.
    """
    from browser_use import Agent  # type: ignore[import-not-found]

    prompt = instruction_template.format(since=since, max_docs=max_docs)
    llm = agent_llm()
    agent = Agent(task=prompt, llm=llm)  # pyright: ignore[reportUnknownVariableType]
    result = await agent.run()  # pyright: ignore[reportUnknownMemberType]
    return parse_agent_result(result, default_doc_type=default_doc_type)


def agent_llm() -> object:
    """Pick ChatAnthropic or ChatOpenAI based on `CIVIC_SLM_LLM_BACKEND`.

    Honors `CIVIC_SLM_STRICT_LOCAL` — under strict-local, an Anthropic-bound
    backend choice raises rather than silently spending tokens.
    """
    from civic_slm.serve.runtimes import is_strict_local

    choice = os.environ.get("CIVIC_SLM_LLM_BACKEND", "anthropic").strip().lower()
    if is_strict_local() and choice != "local":
        raise RuntimeError(
            f"CIVIC_SLM_STRICT_LOCAL is set, but CIVIC_SLM_LLM_BACKEND={choice!r}. "
            "In strict-local mode, the browser-use crawler refuses to use "
            "Anthropic. Set CIVIC_SLM_LLM_BACKEND=local or unset CIVIC_SLM_STRICT_LOCAL."
        )
    if choice == "local":
        from browser_use.llm import ChatOpenAI  # type: ignore[import-not-found]

        return ChatOpenAI(  # pyright: ignore[reportUnknownVariableType]
            model=os.environ.get("CIVIC_SLM_LOCAL_LLM_MODEL", "default"),
            base_url=os.environ.get("CIVIC_SLM_LOCAL_LLM_URL", "http://127.0.0.1:8081") + "/v1",
            api_key="not-needed",
        )
    from browser_use.llm import ChatAnthropic  # type: ignore[import-not-found]

    from civic_slm.config import require

    return ChatAnthropic(  # pyright: ignore[reportUnknownVariableType]
        model="claude-sonnet-4-6",
        api_key=require("ANTHROPIC_API_KEY"),
    )


def parse_agent_result(
    result: object,
    *,
    default_doc_type: DocType = DocType.AGENDA,
) -> list[DiscoveredDoc]:
    """Pull a JSON array out of the agent's text output; be liberal on parse errors."""
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
                doc_type=default_doc_type,
                meeting_date=date if isinstance(date, str) else None,
            )
        )
    return out
