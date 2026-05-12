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
  - `https://www.sanclemente.gov/...` (city website, agendas, staff reports — CivicPlus-hosted)
  - `https://www.youtube.com/@san-clemente-tv/...` (council meeting recordings,
    captions only — audio is fetched only if captions are unavailable)
- **Site terms-of-use:** captured 2026-05-12 from
  [`https://www.sanclemente.gov/copyright`](https://www.sanclemente.gov/copyright):

  > "All content © 2006-2026 San Clemente, CA and its representatives. All rights reserved."
  > "CivicPlus Content Management System © 1997-2026 CivicPlus. All rights reserved."

  The privacy policy at
  [`https://www.sanclemente.gov/site/privacy`](https://www.sanclemente.gov/site/privacy)
  is silent on content reuse. No explicit grant of redistribution,
  derivative-work, or AI-training rights appears anywhere on the public site.

- **robots.txt:** _not yet confirmed; pull and quote before any real crawl._
- **Public-records statute:** California Public Records Act (Gov. Code §7920
  et seq.) and the Ralph M. Brown Act (Gov. Code §54950 et seq.). Council
  meeting agendas, minutes, and staff reports are required to be made
  publicly available **for inspection** — CPRA does not, on its face,
  grant downstream redistribution rights.
- **Audit date:** 2026-05-12
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **NO-GO for redistribution of a derivative training corpus.**
  Crawling for local inspection / smoke testing of the pipeline is fine; what
  the audit blocks is publishing weights or a dataset derived from this
  jurisdiction's content until one of the carve-outs below lands.
- **Rationale:** The city asserts "all rights reserved" in plain language
  and offers no reuse grant in either the copyright or privacy pages. CPRA
  establishes the right to _read_ public records (agendas, minutes, staff
  reports), but the city retains copyright on the documents themselves —
  CPRA does not strip that. Without a reuse grant, a published fine-tune
  trained on this corpus is on shaky ground. Three carve-outs would flip
  this to GO:
  1. **Explicit permission from the City Clerk** — a one-line email
     authorizing use of agendas/staff reports/minutes for an open-source
     research model. Email contact published at
     [`https://www.sanclemente.gov/our-city/contact-us`](https://www.sanclemente.gov/our-city/contact-us).
  2. **Documented fair-use posture from counsel** — argues transformative
     fair use under 17 USC §107 for the training step + public release.
     Not a layperson decision; requires actual legal review and a written
     posture committed to this repo.
  3. **Restrict to genuinely-public-domain material** — federal-government
     content embedded in city docs is `§105` public domain; council
     meeting transcripts of _speech_ (vs. recorded video) may be argued
     uncopyrightable as oral statements in a public forum, but this is
     not settled. Narrowing the recipe to those slices is fragile.

  Until one of (1)/(2)/(3) is in place, the v0.1.0/v0.2.x posture stands:
  the recipe runs against synthetic test fixtures and the maintainer's
  local-only smoke crawls. Published weights or datasets derived from
  San Clemente content require a follow-up GO commit on this entry.

- **Special handling:**
  - PII scrubbing: ON. Public-comment portions of meeting transcripts have
    speaker labels and street addresses scrubbed regardless of env vars.
  - Rate limit: at most 1 request/second to `sanclemente.gov` to avoid
    burdening the site.
  - YouTube captions are preferred over Whisper ASR; when ASR is needed,
    Apple-Silicon `mlx-whisper` runs locally (no third-party transcript
    services).
