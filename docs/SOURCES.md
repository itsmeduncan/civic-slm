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

---

## Maintainer GO posture — v0.2.x diverse-jurisdiction cohort (2026-05-14)

The 8 jurisdictions below (seattle, nyc, boston, denver, portland-or, cook-county, atlanta, austin) are flipped to **Decision: GO** under a blanket maintainer posture so the v1 retrain has corpus diversity beyond san-clemente. Seven of the eight are on Legistar/Granicus; portland-or turned out to be a custom Drupal site (#59) and is included under the same posture because the fair-use analysis is statute-driven, not vendor-driven. This is _not_ the same standard of audit as san-clemente (which has verbatim ToS quotes captured). Specifically:

- Per-site ToS verbatim quotes and robots.txt are **not yet captured.** They are scheduled as a per-jurisdiction follow-up (`docs/SOURCES.md` is the right home — each entry's "Site terms-of-use" / "robots.txt" fields stay marked as not-yet-captured until the maintainer fills them in).
- The fair-use posture is **identical** to san-clemente's: training, evaluation, and publication of derivative model weights on publicly-available U.S. local-government records published under a state-level public-records / open-meetings statute (each entry names the specific statute).
- Rate limits and right-of-withdrawal are inherited from the san-clemente template: 1 request/second, 30 days to remove on request.
- If a specific jurisdiction's ToS turns out to forbid this kind of use, **its entry will be flipped back to NO-GO and the affected corpus removed from `data/raw/` plus any derived `data/processed/` and `data/sft/`** within 30 days. The blanket position is therefore reversible per-jurisdiction.

Filing this preamble keeps the gate honest: GO here is _maintainer's informed judgment under uniform statutory floors_, not _per-site ToS clearance_. A v1.1+ pass should backfill the verbatim ToS quotes for each.

---

## seattle (WA)

- **Recipe file:** `src/civic_slm/ingest/recipes/seattle.yaml`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://seattle.legistar.com/...` (Legistar calendar + meeting detail pages — _verify URL before first crawl_)
  - `https://www.seattle.gov/council/...` (city council pages; some agenda PDFs may resolve here rather than Legistar)
- **Site terms-of-use:** _not yet captured. Pull from the Legistar footer ("This site is hosted by Granicus, LLC...") and the seattle.gov terms before flipping to GO._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Washington Public Records Act (RCW 42.56) and the Washington Open Public Meetings Act (RCW 42.30). City Council agendas, minutes, and supporting materials are public by statute.
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass. Anticipated posture: fair-use parallel to San Clemente, plus Legistar/Granicus-vendor verification (Granicus's hosting terms typically grant municipalities full content rights, but per-tenant clauses vary).
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `seattle.legistar.com`.
  - Right of withdrawal: 30 days from a City of Seattle request.

---

## nyc (NY)

- **Recipe file:** `src/civic_slm/ingest/recipes/nyc.yaml`
- **Jurisdiction type:** city (consolidated, with NYC Council as the legislative body)
- **Source URL patterns:**
  - `https://legistar.council.nyc.gov/...` (Council Legistar — _verify URL before first crawl; the Council reshuffled its host in the past_)
  - `https://council.nyc.gov/...` (Council content pages)
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** New York Freedom of Information Law (NY Public Officers Law Art. 6) and the Open Meetings Law (NY Public Officers Law Art. 7). Council legislative records are public.
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass.
- **Special handling:**
  - PII scrubbing: ON. NYC public comment is high-volume and frequently includes residents' street addresses and personal histories — scrubbing must remain on.
  - Rate limit: 1 request/second to `legistar.council.nyc.gov`.
  - Right of withdrawal: 30 days from a NYC Council request.

---

## boston (MA)

- **Recipe file:** `src/civic_slm/ingest/recipes/boston.yaml`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://boston.legistar.com/...` — _verify URL before first crawl._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Massachusetts Public Records Law (M.G.L. c. 66, §10) and the Open Meeting Law (M.G.L. c. 30A, §§18-25).
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `boston.legistar.com`.
  - Right of withdrawal: 30 days from a City of Boston request.

---

## denver (CO)

- **Recipe file:** `src/civic_slm/ingest/recipes/denver.yaml`
- **Jurisdiction type:** city + county (consolidated)
- **Source URL patterns:**
  - `https://denver.legistar.com/...` — _verify URL before first crawl._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Colorado Open Records Act (CRS §24-72-201 et seq.) and the Colorado Open Meetings Law (CRS §24-6-401).
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `denver.legistar.com`.
  - Right of withdrawal: 30 days from a City of Denver request.

---

## portland-or (OR)

- **Recipe file:** `src/civic_slm/ingest/recipes/portland-or.yaml`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://www.portland.gov/council/agenda/...` — Drupal-rendered agenda index + per-meeting pages.
  - `https://efiles.portlandoregon.gov/record/<id>/file/document` — agenda PDFs.
  - _Not on Legistar despite a placeholder `portland.legistar.com` tenant existing — see #59. Slug includes `-or` to disambiguate from Portland ME, a likely future recipe._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Oregon Public Records Law (ORS 192.311 et seq.) and the Oregon Public Meetings Law (ORS 192.610 et seq.).
- **Audit date:** 2026-05-15
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x maintainer posture documented above (cohort is "diverse jurisdictions," not "Legistar specifically" — Portland uses a custom Drupal site).
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `www.portland.gov` and `efiles.portlandoregon.gov`.
  - Right of withdrawal: 30 days from a City of Portland (OR) request.

---

## cook-county (IL)

- **Recipe file:** `src/civic_slm/ingest/recipes/cook-county.yaml`
- **Jurisdiction type:** county
- **Source URL patterns:**
  - `https://cook-county.legistar.com/...` (Board of Commissioners + Forest Preserve + committees) — _verify URL before first crawl._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Illinois Freedom of Information Act (5 ILCS 140) and the Illinois Open Meetings Act (5 ILCS 120). County legislative records are public.
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass. First **county** recipe — verify that the recipe template's "skip committees" heuristic doesn't accidentally drop Cook County Finance / Zoning / Health committees that publish substantive agendas.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `cook-county.legistar.com`.
  - Right of withdrawal: 30 days from a Cook County request.

---

## atlanta (GA)

- **Recipe file:** `src/civic_slm/ingest/recipes/atlanta.yaml`
- **Jurisdiction type:** city
- **Source URL patterns:**
  - `https://atlanta.legistar.com/...` — _verify URL before first crawl._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Georgia Open Records Act (O.C.G.A. §50-18-70 et seq.) and the Georgia Open Meetings Act (O.C.G.A. §50-14-1 et seq.).
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `atlanta.legistar.com`.
  - Right of withdrawal: 30 days from a City of Atlanta request.

---

## austin (TX)

- **Recipe file:** `src/civic_slm/ingest/recipes/austin.yaml`
- **Jurisdiction type:** city (home-rule under Texas Local Government Code)
- **Source URL patterns:**
  - `https://austintexas.legistar.com/...` — _verify URL before first crawl._
- **Site terms-of-use:** _not yet captured._
- **robots.txt:** _not yet confirmed._
- **Public-records statute:** Texas Public Information Act (Gov't Code Ch. 552) and the Texas Open Meetings Act (Gov't Code Ch. 551). Home-rule cities still operate under these statutes.
- **Audit date:** 2026-05-14
- **Auditor:** itsmeduncan (project maintainer)
- **Decision:** **GO** under the v0.2.x Legistar-cohort maintainer posture documented above.
- **Rationale:** Fair-use parallel to san-clemente, applied to the named state public-records statute. Per-site ToS verbatim quotes are deferred to a v1.1+ pass. SUP / TIRZ vocabulary in the Austin agendas is part of why the v0.2 eval bench specifically seeded Texas examples; this recipe is what unblocks training on real Texas docs.
- **Special handling:**
  - PII scrubbing: ON.
  - Rate limit: 1 request/second to `austintexas.legistar.com`.
  - Right of withdrawal: 30 days from a City of Austin request.
