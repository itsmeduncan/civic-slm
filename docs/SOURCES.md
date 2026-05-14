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
- **Decision:** **GO.** Maintainer override: training, evaluation, and
  publication of derivative model weights proceed under a fair-use posture
  for transformative research use of publicly-available civic records.
- **Rationale:** Records published under CPRA and the Brown Act are public
  by statute. Training a small civic-NLP model on them, and releasing the
  resulting weights with attribution, is treated here as transformative
  fair use under 17 USC §107: the model produces a new work (a research
  artifact) that does not substitute for the original documents in any
  market. If the City asks for content to be withdrawn, the project
  honors that within 30 days — see "Special handling" below.

- **Special handling:**
  - PII scrubbing: ON. Public-comment portions of meeting transcripts have
    speaker labels and street addresses scrubbed regardless of env vars.
  - Rate limit: at most 1 request/second to `sanclemente.gov` to avoid
    burdening the site.
  - YouTube captions are preferred over Whisper ASR; when ASR is needed,
    Apple-Silicon `mlx-whisper` runs locally (no third-party transcript
    services).
  - Right of withdrawal: if the City requests removal of any content, the
    affected material is dropped from subsequent releases within 30 days.

---

## santa-monica (CA)

- **Recipe file:** `src/civic_slm/ingest/recipes/santa-monica.yaml`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://santamonicacityca.iqm2.com/Citizens/...` (IQM2 / iCompass meeting calendar — _verify before first crawl; vendor may have rebranded_)
  - YouTube: `https://www.youtube.com/@cityofsantamonica` (currently commented out in the YAML — wire when issue #45's `discover_videos` path lands)
- **Site terms-of-use:** _not yet captured. Pull the relevant clause from the IQM2-hosted Citizen Portal footer and the santamonica.gov terms before flipping to GO._
- **robots.txt:** _not yet confirmed. Pull and quote before any real crawl._
- **Public-records statute:** California Public Records Act (Gov. Code §7920 et seq.) and the Ralph M. Brown Act (Gov. Code §54950 et seq.). Same statutory floor as San Clemente.
- **Audit date:** _PENDING_
- **Auditor:** _PENDING_
- **Decision:** **PENDING.** Recipe file lands in this PR; the ToS audit + Decision flip is a separate maintainer task — see #18 / #37 for how San Clemente was handled.
- **Rationale:** _To be filled at audit time. Anticipated posture: same fair-use stance as San Clemente, plus IQM2-specific verification that the vendor's hosting terms don't preempt the City's public-records duty._
- **Special handling:**
  - PII scrubbing: ON (same default as San Clemente).
  - Rate limit: 1 request/second to `santamonicacityca.iqm2.com`.
  - Right of withdrawal: 30 days from a City of Santa Monica request.
