# Adding a new jurisdiction

This project is built to crawl any U.S. city, county, or township that publishes meeting documents on the open web. San Clemente, CA ships as the demo recipe. Adding more jurisdictions is a copy-paste job — usually under 30 lines of code per jurisdiction.

> **California-isms are not universal.** The reference recipe and the
> v0 eval contexts are San-Clemente-styled — they use CEQA exemptions,
> CUP file numbers, and California General Plan land-use vocabulary.
> Texas, New York, and Ohio jurisdictions do these very differently:
> Texas uses Specific Use Permits instead of CUPs; NY uses SEQRA in
> place of CEQA; many states call their long-range planning document
> a "comprehensive plan" or "master plan" rather than a "general
> plan." When you write a recipe for a non-California jurisdiction,
> do not paste over the San-Clemente-flavored prompts and field
> names — adjust them to the local vocabulary. The schema is
> deliberately neutral (`general_plan | comprehensive_plan |
master_plan` are all accepted by `DocType`); the prompts are not.
> See `docs/GLOSSARY.md` for the civic-vocabulary glossary that
> non-California recipes should align with.

## What a recipe is

A recipe is a Python dataclass that satisfies the `Recipe` Protocol from `src/civic_slm/ingest/harness.py`:

```python
class Recipe(Protocol):
    @property
    def jurisdiction(self) -> str: ...   # kebab-case slug, e.g. "harris-county"
    @property
    def state(self) -> str: ...          # 2-letter postal code, e.g. "TX"
    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]: ...
```

`discover` runs an LLM-driven browser agent against the jurisdiction's website and returns a list of documents to fetch. The crawler's orchestrator (`harness.crawl()`) handles the rest: HTTP fetch, sha256 dedupe, text extraction, manifest append.

## Step 1 — Copy the template

```bash
cp src/civic_slm/ingest/recipes/_template.py \
   src/civic_slm/ingest/recipes/austin.py     # for example
```

## Step 2 — Edit three things

In your new file:

1. **Class name and slug + state.**

   ```python
   @dataclass(frozen=True)
   class AustinRecipe:
       jurisdiction: str = "austin"
       state: str = "TX"
       instruction: str = INSTRUCTION
   ```

2. **The `INSTRUCTION` string.** This is what the browser-use agent reads. Be specific about the start URL, what to look for (in user-visible terms), what to skip, and the JSON output shape. Keep it under ~250 tokens — long instructions confuse the agent.

   ```text
   Open https://www.austintexas.gov/department/city-council and navigate to
   the city council meetings calendar. Find the list of past regular City
   Council meetings going back to {since}, up to {max_docs} most recent.
   For each meeting, return:
     - title: the meeting name
     - meeting_date: the date in YYYY-MM-DD format
     - source_url: the direct URL to the PDF or HTML agenda
   Skip work sessions, executive sessions, and special-called meetings.
   Return strictly as a JSON array of objects with the three keys above.
   ```

3. **Leave `discover` and `_parse_result` alone.** Both are copy-paste boilerplate; you only touch them if your jurisdiction has an unusual output shape (e.g. requires login, paginated lists across many pages).

## Step 3 — Register the recipe

Open `src/civic_slm/ingest/crawl.py` and add the recipe to `_RECIPES`:

```python
from civic_slm.ingest.recipes.austin import AustinRecipe
from civic_slm.ingest.recipes.san_clemente import SanClementeRecipe

_RECIPES: dict[str, Callable[[], Recipe]] = {
    "san-clemente": SanClementeRecipe,
    "austin": AustinRecipe,
}
```

## Step 4 — Try it

```bash
uv run civic-slm crawl --jurisdiction austin --max 5
```

Watch the agent navigate. If it returns an empty list, the most common issues are:

- Start URL is wrong or behind a redirect that confuses the agent.
- Page requires JavaScript that browser-use isn't rendering — try `playwright install chromium` if you skipped it.
- Selectors changed; the agent gives up and returns `[]`. Tighten the `INSTRUCTION` with a specific link text or page title to anchor on.
- Captcha or login wall. We don't bypass these; flag the jurisdiction as unsupported.

## Step 5 — Verify the manifest

```bash
tail -5 data/raw/manifest.jsonl
```

Each line should have `state: "TX"`, `jurisdiction: "austin"`, a real `source_url`, and a non-empty `text` field after PDF extraction.

## Tips for hard jurisdictions

- **Granicus / Legistar / CivicPlus / IQM / PrimeGov / Municode** — all of these have predictable patterns. The browser-use agent generally handles them with a one-line instruction. If yours is broken, look at how the San Clemente recipe phrases things.
- **Custom CMS, weird PDF viewer.** Be explicit in the instruction: "On the meeting detail page, click the 'Agenda Packet' link to find the PDF URL."
- **Counties / townships / school districts.** Same shape — the `jurisdiction` slug just becomes `harris-county`, `montgomery-township`, `nyc-doe`. The recipe doesn't care what level of government it is.
- **Multiple jurisdictions same name.** If two states have a "Springfield," disambiguate with the slug: `springfield-il` vs `springfield-mo`.

## Naming convention

- `jurisdiction` slug: kebab-case, lowercase, ASCII only. Strip suffixes like "city", "town", "county" unless needed for disambiguation.
- `state`: uppercase 2-letter USPS postal code (`CA`, `TX`, `NY`, `DC`).
- File name: `<jurisdiction>.py` matching the slug.
- Class name: `<TitleCase>Recipe`, e.g. `HarrisCountyRecipe`.

## Adding video sources

Some jurisdictions stream meetings to a YouTube channel. To pull those into your corpus alongside text documents, add a `discover_videos` method to your recipe.

```python
# src/civic_slm/ingest/recipes/austin.py (or wherever)
from civic_slm.ingest.harness import DiscoveredVideo
from civic_slm.ingest.recipes._youtube import youtube_channel_videos


@dataclass(frozen=True)
class AustinRecipe:
    jurisdiction: str = "austin"
    state: str = "TX"
    instruction: str = INSTRUCTION
    youtube_channel: str = "https://www.youtube.com/@CityofAustin/videos"

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        ...   # PDFs as before

    async def discover_videos(self, *, since: str, max_videos: int) -> list[DiscoveredVideo]:
        return youtube_channel_videos(
            self.youtube_channel, since=since, max_videos=max_videos
        )
```

Then run:

```bash
civic-slm crawl-videos --jurisdiction austin --since 2025-01-01 --max 20
```

What happens: `yt-dlp` enumerates the channel, downloads `bestaudio` as `.m4a`, and writes both the human-uploaded and auto-generated VTT captions. The transcript orchestrator picks the best available source — human SRT/VTT → auto-caption → Whisper ASR fallback. The resulting transcript text lands in the manifest as a `meeting_transcript` doc, indistinguishable downstream from any other document.

Whisper ASR is ~1× real-time on Apple Silicon (a 3-hour council meeting → ~3 hours). Most public-meeting channels publish auto-captions, so the Whisper path is rare in practice.

`docs/RUNTIMES.md` doesn't apply here — video ingestion runs locally via `yt-dlp` + (optionally) `mlx-whisper`, not via the OpenAI-compatible chat runtime layer.

## A note on strict-local mode

Both `recipes/_browser.py` (the helper used by `_template.py` and `san_clemente.py`) and `recipes/_youtube.py` (the helper used for video discovery) honor `CIVIC_SLM_STRICT_LOCAL=1` automatically — under strict-local, the browser-use agent refuses an Anthropic-bound configuration and raises before any tokens ship.

If you write a recipe that **instantiates `ChatAnthropic` or any other paid SDK directly** (instead of using `run_browser_agent()`), gate it on `civic_slm.serve.runtimes.is_strict_local()` so it raises in the same conditions:

```python
from civic_slm.serve.runtimes import is_strict_local

if is_strict_local() and using_anthropic:
    raise RuntimeError(
        "CIVIC_SLM_STRICT_LOCAL is set; recipe refuses to use Anthropic."
    )
```

Otherwise your recipe is a hole in the tripwire.

## What lands on disk

For each crawled doc:

```
data/raw/<state-lower>/<jurisdiction>/<meeting-date>/<safe-title>-<sha8>.<ext>
data/raw/manifest.jsonl                                # one CivicDocument per line
```

The manifest is the audit trail. It's committed to the repo (the raw bytes are gitignored). Re-running the crawl is idempotent — sha256 dedupe means existing docs are skipped silently.
