# Adding a new jurisdiction

This project crawls any U.S. city, county, or township that publishes
meeting documents on the open web. The fast path is **a 10-line YAML
file**. Drop it in `src/civic_slm/ingest/recipes/`, the crawler picks it
up. No `_RECIPES` dict to edit, no Python required for the common case.

> **California-isms are not universal.** The reference recipe and the
> v0 eval contexts are San-Clemente-styled — they use CEQA exemptions,
> CUP file numbers, and California General Plan land-use vocabulary.
> Texas, New York, and Ohio jurisdictions do these very differently:
> Texas uses Specific Use Permits instead of CUPs; NY uses SEQRA in
> place of CEQA; many states call their long-range planning document
> a "comprehensive plan" or "master plan" rather than a "general
> plan." The `DocType` schema already accepts the regional variants
> (`general_plan | comprehensive_plan | master_plan`); the vendor
> prompt templates do not yet — adjust your YAML's `instruction:`
> override if your jurisdiction's site exposes non-California
> vocabulary in the navigation. See `docs/GLOSSARY.md` for the
> civic-vocabulary reference.

## The 5-minute path (YAML)

```bash
uv run civic-slm new-recipe
# prompts: slug → state → vendor → start URL → (optional) YouTube channel
# writes:   src/civic_slm/ingest/recipes/<slug>.yaml

# Add a docs/SOURCES.md stub for <slug> (Decision: PENDING is fine).
# A maintainer flips Decision: GO after reviewing the site's ToS.

uv run civic-slm crawl <slug> --max 5 --since 2025-09-01
wc -l data/raw/manifest.jsonl   # +5 new lines, your jurisdiction
```

The generated YAML looks like:

```yaml
jurisdiction: santa-monica
state: CA
vendor: iqm2
start_url: https://santamonicacityca.iqm2.com/Citizens/Calendar.aspx
doc_type_default: agenda
youtube_channel: https://www.youtube.com/@cityofsantamonica # optional
```

`vendor:` selects a prompt template from
`src/civic_slm/ingest/recipes/_vendors/` that tells the browser-use
agent what to look for on that platform. `start_url:` is the only piece
of jurisdiction-specific knowledge you need to provide. The template
covers the rest.

## Vendor cheat sheet

| Vendor                 | Recipe `vendor:` value | Typical start URL                                   |
| ---------------------- | ---------------------- | --------------------------------------------------- |
| CivicPlus AgendaCenter | `civicplus`            | `https://<city>.gov/AgendaCenter`                   |
| IQM2 / iCompass        | `iqm2`                 | `https://<city>.iqm2.com/Citizens/Calendar.aspx`    |
| Granicus               | `granicus`             | `https://<city>.granicus.com/...`                   |
| Legistar               | `legistar`             | `https://<jurisdiction>.legistar.com/Calendar.aspx` |
| PrimeGov               | `primegov`             | `https://<city>.primegov.com/Public/Calendar`       |
| Municode               | `municode`             | _varies — check the City's "Meetings" page first_   |

Granicus and Legistar share one template (`granicus_legistar.md`);
PrimeGov and Municode share another (`primegov_municode.md`,
experimental). The aliases `granicus` / `legistar` and `primegov` /
`municode` resolve to the same template — pick whichever matches the
URL you actually see.

**Don't trust `<city>.legistar.com` existing as proof the city is on
Legistar.** Several jurisdictions have placeholder Legistar tenants
that return "Invalid parameters!" on `Calendar.aspx` and "not set up"
on `webapi.legistar.com/v1/<city>/...`. Verify a real meeting row
renders before assuming the vendor. Known examples:

- **Portland, OR** — actual calendar is at
  `https://www.portland.gov/council/agenda/all` (Drupal); PDFs live at
  `efiles.portlandoregon.gov`. See `portland-or.yaml` for the
  `instruction:` override pattern. (#59)

If you know your city has a custom site or none of the above fit:
provide an `instruction:` override in your YAML with the prompt the
agent should run instead. The full text replaces the vendor template.

## Step zero — the ToS audit

Every recipe needs an entry in `docs/SOURCES.md`. A recipe PR without a
SOURCES.md entry is rejected at review.

Copy the template from the top of `docs/SOURCES.md` and fill in:

- the source URL patterns
- a verbatim quote from the site's terms-of-use
- the public-records statute that covers the content
- a Decision: **GO** or **NO-GO**

A first-pass PR with **Decision: PENDING** is fine — the YAML can land
alongside the audit stub, and a maintainer flips Decision: GO once the
audit is reviewed. The crawler does not enforce this at runtime; it is
a documentation contract. Don't run a real crawl against a NO-GO /
PENDING jurisdiction.

## Adding video sources

Some jurisdictions stream meetings to a YouTube channel. Set
`youtube_channel:` in your YAML and `civic-slm crawl-videos <slug>`
will enumerate via `yt-dlp`, prefer captions, and fall back to
Whisper ASR locally if no captions are published.

```yaml
youtube_channel: https://www.youtube.com/@cityofsantamonica
```

The transcript text lands in the manifest as a `meeting_transcript`
doc, indistinguishable downstream from any PDF agenda. Whisper ASR is
~1× real-time on Apple Silicon, so a 3-hour council meeting transcribes
in ~3 hours. Most public-meeting channels publish auto-captions, so the
Whisper path is rare in practice.

## Custom logic — the Python escape hatch

When YAML isn't enough — a vendor needs login, a non-browser API call,
a custom JSON shape — drop down to Python:

```bash
cp src/civic_slm/ingest/recipes/_template.py \
   src/civic_slm/ingest/recipes/austin.py
```

Edit three things: the class name + slug + state, the `INSTRUCTION`
string, and (optionally) the `discover_videos` method. The crawler
auto-discovers `*.py` files in the recipes directory alongside `*.yaml`,
so no registration step is needed. See `recipes/san_clemente.py` for
the reference Python recipe.

A custom Python recipe satisfies the same `Recipe` Protocol from
`src/civic_slm/ingest/harness.py`:

```python
class Recipe(Protocol):
    @property
    def jurisdiction(self) -> str: ...   # kebab-case slug, e.g. "harris-county"
    @property
    def state(self) -> str: ...          # 2-letter postal code, e.g. "TX"
    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]: ...
```

When a `.py` and `.yaml` define the same slug, the YAML wins (with a
log line). That lets you stage a YAML migration on top of a working
Python recipe without losing the Python fallback if you need it.

## When the crawl returns nothing

```bash
uv run civic-slm crawl <slug> --max 5 --since 2025-09-01
# discovered_docs count=0
```

Most common causes:

- **Wrong start URL.** Open the URL in a browser yourself; if it's a
  redirect, follow it and update the YAML.
- **JavaScript-only nav.** `playwright install chromium` if you skipped it.
- **Site selectors changed.** Browser-use is robust to most rewrites
  but loses on heavily-redesigned vendor pages. Override `instruction:`
  with a more specific prompt anchoring on visible link text.
- **Captcha or login wall.** The project does not bypass these; flag
  the jurisdiction as unsupported in `docs/SOURCES.md` and move on.
- **Vendor mismatch.** PrimeGov sites occasionally look like Granicus
  archives at first glance. Try the other template.

## Currently registered

YAML recipes shipped in `src/civic_slm/ingest/recipes/`. **All are gated PENDING in `docs/SOURCES.md`** — a maintainer flips Decision: GO after auditing each jurisdiction's ToS, robots.txt, and statutory floor. Until then, `civic-slm crawl <slug>` will resolve the recipe but the audit gate blocks the real crawl.

| Slug           | State | Vendor    | Jurisdiction type     | Region          |
| -------------- | ----- | --------- | --------------------- | --------------- |
| `san-clemente` | CA    | civicplus | city                  | Southwest (Pac) |
| `santa-monica` | CA    | iqm2      | city                  | Southwest (Pac) |
| `seattle`      | WA    | legistar  | city                  | Pacific NW      |
| `portland-or`  | OR    | custom    | city                  | Pacific NW      |
| `denver`       | CO    | legistar  | city + county (cons.) | Mountain West   |
| `cook-county`  | IL    | legistar  | county                | Midwest         |
| `austin`       | TX    | legistar  | city (home-rule)      | South Central   |
| `atlanta`      | GA    | legistar  | city                  | South           |
| `boston`       | MA    | legistar  | city                  | Northeast       |
| `nyc`          | NY    | legistar  | city (consolidated)   | Northeast       |

Platform coverage: CivicPlus (1), IQM2 (1), Legistar/Granicus (7), custom (1 — `portland-or`, see #59). **Municode and PrimeGov are not yet represented** — future contributions welcome; both share the `primegov` / `municode` vendor template alias.

## Naming conventions

- `jurisdiction` slug: kebab-case, lowercase, ASCII only. Strip suffixes
  like "city", "town", "county" unless needed for disambiguation.
- `state`: uppercase 2-letter USPS postal code (`CA`, `TX`, `NY`, `DC`).
- File name: `<slug>.yaml` (or `<slug>.py` for Python) matching the slug.
- Python class name (escape-hatch path only): `<TitleCase>Recipe`, e.g.
  `HarrisCountyRecipe`.

If two states share a jurisdiction name, disambiguate in the slug:
`springfield-il` vs `springfield-mo`.

## Strict-local mode

`recipes/_browser.py` and `recipes/_youtube.py` honor
`CIVIC_SLM_STRICT_LOCAL=1` automatically — under strict-local, the
browser-use agent refuses an Anthropic-bound configuration and raises
before any tokens ship. The vendor templates flow through these
helpers, so YAML recipes inherit the tripwire.

If a Python recipe instantiates `ChatAnthropic` or any other paid SDK
**directly** (instead of using `run_browser_agent()`), gate it on
`civic_slm.serve.runtimes.is_strict_local()`:

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

The manifest is the audit trail and is committed to the repo (raw
bytes are gitignored). Re-running the crawl is idempotent — sha256
dedupe means existing docs are skipped silently.
