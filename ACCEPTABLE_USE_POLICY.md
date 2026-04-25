# Acceptable Use Policy

This policy applies to anyone using `civic-slm` code, weights, datasets, or
derivative works.

## Intended use

`civic-slm` is built to help **civic technologists, journalists, public
servants, and residents understand publicly-available local-government
documents** — agendas, staff reports, meeting minutes, ordinances,
comprehensive plans, and municipal codes. Examples of supported use:

- Summarizing a 200-page general plan for a non-specialist audience.
- Extracting structured fields from a staff report.
- Helping a resident find where in a municipal code a particular topic is addressed.
- Powering search and Q&A in a public-records transparency tool, with citations.

## Out of scope — do not use this model for

- **Legal advice.** This model is not a lawyer. It is not authorized to give
  individualized legal opinions on zoning, permitting, code compliance, or
  any other matter.
- **Voter eligibility, ballot, or election determinations.** Do not use the
  model to tell anyone whether they are eligible to vote, what is on their
  ballot, or how to vote.
- **Benefits eligibility.** Do not use the model to determine whether
  someone qualifies for housing assistance, social services, or other
  government benefits.
- **High-stakes decisions about individuals** named in scraped civic
  documents — including but not limited to: identifying public commenters,
  inferring opinions of named officials, profiling residents who appear in
  meeting transcripts.
- **Surveillance, doxxing, or targeted harassment** of public commenters,
  staff, or elected officials whose names appear in scraped documents.
- **Generating fake civic content** — fake meeting minutes, fake staff
  reports, fake ordinances — for distribution as if real.
- **Automated submission** to public-comment portals, permitting systems,
  or other government interfaces in a way that bypasses the human-review
  requirements of those systems.

## Required disclosures when redistributing

If you build a product on top of this model:

1. Disclose to end-users that responses come from a language model and may be
   wrong.
2. When the model produces a citation, link the user to the source document.
3. Do not present the model's output as the position of any city, county, or
   official.

## Geographic and temporal limitations

The training corpus is biased toward U.S. local-government documents, with the
demo recipe focused on San Clemente, CA. The model:

- Has not been evaluated on jurisdictions outside the United States.
- Has not been evaluated on tribal-government or federal documents.
- Has a hard knowledge cutoff at the date of the most recent training-data
  crawl (see `MODEL_CARD.md`). It does not know about ordinances, votes, or
  staff appointments that happened after the cutoff.

## Reporting violations

If you see a deployment of `civic-slm` that violates this policy, please
report it to `itsmeduncan@gmail.com` with subject prefix
`[civic-slm AUP]`. We will publicly disclose patterns of misuse without
identifying individual reporters.

## Enforcement

This AUP is a community norm. The MIT-licensed code does not legally bind
downstream users to it, but contributions to this repository are conditioned
on agreement to abide by it, and forks that violate it will not receive
upstream support, security advisories, or model-weight access.
