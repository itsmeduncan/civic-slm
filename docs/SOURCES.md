# Source-License Audit

This file is the **gate** for ingesting any real civic document into
`data/raw/`. Each registered recipe in `src/civic_slm/ingest/recipes/`
must have an entry here, filled in by the recipe author and reviewed by
a maintainer before its first crawl.

A new recipe PR that does not include an entry here is rejected.

## Audit template (copy this for each recipe)

```
## <jurisdiction-slug> (<STATE>)

- **Recipe file:** `src/civic_slm/ingest/recipes/<slug>.py`
- **Jurisdiction type:** city | county | township | school district
- **Source URL patterns:**
  - `https://example.gov/agendas/...`
- **Site terms-of-use:** <URL>
  - Relevant clause (verbatim quote): "..."
- **robots.txt:** <URL>
  - Relevant directives: ...
- **Public-records statute:** California Public Records Act / Brown Act / etc.
- **Audit date:** YYYY-MM-DD
- **Auditor:** <github handle>
- **Decision:** GO | NO-GO
- **Rationale (one paragraph):** ...
- **Special handling:** PII scrubbing on/off; rate limit; allowed crawl windows.
```

---

## san-clemente (CA)

- **Recipe file:** `src/civic_slm/ingest/recipes/san_clemente.py`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://www.san-clemente.org/...` (city website, agendas, staff reports)
  - `https://www.youtube.com/@san-clemente-tv/...` (council meeting recordings,
    captions only — audio is fetched only if captions are unavailable)
- **Site terms-of-use:** _to be confirmed by maintainer before first crawl._
  - Action item: capture the verbatim "Reuse" / "Terms" clause from the city
    website footer and quote it here. Until done, the recipe is **NO-GO** for
    real ingestion.
- **robots.txt:** _to be confirmed._
- **Public-records statute:** California Public Records Act (Gov. Code §7920
  et seq.) and the Ralph M. Brown Act (Gov. Code §54950 et seq.). Council
  meeting agendas, minutes, and staff reports are required to be made
  publicly available.
- **Audit date:** _pending._
- **Auditor:** _pending._
- **Decision:** **NO-GO until terms-of-use clause is captured and quoted
  above.** This is intentional: v0.1.0 ships infrastructure without
  ingesting real documents.
- **Rationale:** California public-records statutes make these documents
  publicly inspectable. That establishes the right to _read_; redistribution
  in a derivative LLM training corpus requires confirming the city's site
  terms do not forbid it. Until the terms quote lands, the recipe runs only
  against synthetic test fixtures.
- **Special handling:**
  - PII scrubbing: ON. Public-comment portions of meeting transcripts have
    speaker labels and street addresses scrubbed regardless of env vars.
  - Rate limit: at most 1 request/second to `san-clemente.org` to avoid
    burdening the site.
  - YouTube captions are preferred over Whisper ASR; when ASR is needed,
    Apple-Silicon `mlx-whisper` runs locally (no third-party transcript
    services).
