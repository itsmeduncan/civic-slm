# Permission request — City of San Clemente

This is a draft email asking the City of San Clemente for explicit
permission to use its publicly-available documents (agendas, staff
reports, minutes, municipal code) in a publicly-released, open-source
research model. Send when the maintainer is ready; do not edit in
place without recording who sent it and when.

The audit that motivates this ask lives at
[`docs/SOURCES.md#san-clemente-ca`](../SOURCES.md#san-clemente-ca).

Until the city responds (or until counsel commits a fair-use posture
to this repo), the recipe runs against local smoke crawls only; no
weights or datasets derived from San Clemente content are published.

## Addresses

The city's public directory lists phone numbers but not direct email
addresses. The maintainer's options to obtain the right addresses
before sending:

1. **Call the City Clerk** (949-361-8200) and ask for the
   correspondence email for the Clerk, City Manager, and City Attorney.
2. **Submit the website's contact form**
   (https://www.sanclemente.gov/FormCenter/General-Forms-4/Website-Feedback-73)
   asking for a public-records / correspondence email.

Once obtained, fill in the placeholders below.

## Email — copy from here

> **To:** {City Clerk's office email}
> **Cc:** {City Manager's office email}; {City Attorney email};
> {Communications / Public Information email}
> **From:** Duncan Grazier <itsmeduncan@gmail.com>
> **Subject:** Permission to use San Clemente public documents in an open-source civic AI research model

Dear City Clerk,

I'm writing to request explicit permission to include the City of San
Clemente's publicly-available documents — City Council agendas, staff
reports, meeting minutes, and the municipal code — in an open-source
research project I maintain.

**About the project.** _civic-slm_ is a small, open-source language
model fine-tuned on U.S. local-government documents so that journalists,
civic-tech volunteers, and residents can ask grounded questions of city
records on their own laptops, without sending those questions to a
third-party AI service. The source code lives at
https://github.com/itsmeduncan/civic-slm. San Clemente is the demo
jurisdiction the project was built around because of the city's
already-strong public-documents posture.

**What I'm asking for.** A non-exclusive, royalty-free, perpetual,
revocable grant to:

1. **Crawl** documents the city has already published on
   `sanclemente.gov` and the @san-clemente-tv YouTube channel, subject
   to a rate limit of one request per second.
2. **Process** those documents into a training corpus (text extraction,
   chunking) and a held-out evaluation set.
3. **Train** an AI model on the corpus, locally on Apple Silicon.
4. **Publish** the resulting model weights and the held-out evaluation
   data under an open-source license (Apache-2.0 or OpenRAIL-M), with
   the model card crediting the City of San Clemente as a source
   jurisdiction.

**What we will not do, regardless of any grant.**

- Re-publish your full documents themselves as a standalone dataset.
  Only the trained model and a small held-out evaluation set
  (~200 question/answer pairs) would be released.
- Use the documents to identify or surveil named individuals. Public
  comment is automatically scrubbed of speaker labels and addresses
  before training. The full acceptable-use policy is at
  https://github.com/itsmeduncan/civic-slm/blob/main/ACCEPTABLE_USE_POLICY.md.
- Misrepresent the model as the City's product or position.

**What the City would get.**

- **Attribution** in the model card and any release announcement.
- **A free, locally-runnable tool** the city's own staff and residents
  can use to navigate municipal records, with no per-question cost and
  no data leaving the device.
- **Right of withdrawal:** if the City asks us to stop including its
  content, we will remove San Clemente material from any subsequent
  release within 30 days of written notice.
- **Audit trail:** every document used in training is logged with its
  source URL, retrieval timestamp, and a content hash. We can provide
  the full manifest on request.

**Proposed response.** The simplest path forward, if the City is
willing, is a one-line reply such as: _"The City of San Clemente grants
the civic-slm project permission to use publicly-available agendas,
staff reports, minutes, and municipal-code text for the purposes
described in this email."_ If the City would prefer a more formal
written license, I am happy to work with the City Attorney to draft
terms.

I am also happy to walk through the project over a brief call.

Thank you for your time, and for the City's commitment to open
government — the documents your office publishes are the reason this
project is possible.

Sincerely,

Duncan Grazier
San Clemente resident
itsmeduncan@gmail.com
https://github.com/itsmeduncan/civic-slm

---

## After sending — checklist

- [ ] Record the send date and recipient list in a follow-up commit on
      this file (or in `docs/SOURCES.md`).
- [ ] Add a calendar reminder to follow up if no response in 14 days.
- [ ] When a response arrives:
  - **GO:** quote the verbatim reply in `docs/SOURCES.md#san-clemente-ca`,
    flip Decision to GO with the response date, and link to this email.
  - **NO:** record the denial in the same place; pivot to one of the
    other carve-outs (counsel posture, public-domain slices) or to a
    different jurisdiction.
  - **Silence after 30+ days:** is not a grant. Default posture remains
    NO-GO.
