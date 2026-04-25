# Data Card — civic-slm

> **Status as of v0.1.0:** _infrastructure preview._ The committed evaluation
> set is real and described below. The training corpus is not yet generated;
> this card describes the **contract** the corpus will follow once produced.
> The per-source license audit is the gate for ingesting any real document.

## Datasets in this repository

| Path                                    | Purpose                     | Status (v0.1.0)       | Schema                              |
| --------------------------------------- | --------------------------- | --------------------- | ----------------------------------- |
| `data/eval/civic_factuality.jsonl`      | held-out grounded Q&A       | 10 records, synthetic | `EvalExample` / `FactualityExample` |
| `data/eval/refusal.jsonl`               | refusal + over-refusal      | 14 records, synthetic | `EvalExample` / `RefusalExample`    |
| `data/eval/structured_extraction.jsonl` | JSON extraction             | 5 records, synthetic  | `EvalExample` / `ExtractionExample` |
| `data/eval/side_by_side.jsonl`          | LLM-judged comparisons      | 10 records, synthetic | `EvalExample` / `SideBySideExample` |
| `data/raw/`                             | crawled source documents    | empty (`.gitkeep`)    | `CivicDocument` (`schema.py`)       |
| `data/processed/`                       | cleaned, chunked            | empty (`.gitkeep`)    | `DocumentChunk` (`schema.py`)       |
| `data/sft/`                             | synthetic instruction pairs | empty (`.gitkeep`)    | `InstructionExample` (`schema.py`)  |
| `data/dpo/`                             | preference pairs            | empty (`.gitkeep`)    | `PreferencePair` (`schema.py`)      |

## Eval datasets

### `civic_factuality.jsonl` (10 records)

- **Source:** synthetic; written by hand to mirror common San-Clemente-styled
  staff-report and agenda content. No real public commenters, no real
  addresses, no real names. Personae and businesses are invented.
- **Schema fields:** `id`, `bench`, `question`, `context`, `gold_answer`,
  `gold_citations` (list of strings that must appear verbatim in the model
  output).
- **License:** CC0 / public domain. Hand-written by the maintainers.
- **Known limitations:** California-flavored vocabulary (CUP, CEQA),
  word-overlap scorer.

### `refusal.jsonl` (14 records: 10 should-refuse + 4 should-answer)

- **Source:** synthetic, hand-written.
- **Schema fields:** `id`, `bench`, `question`, `context`, `expected_refusal`
  (boolean).
- **Why it has a should-answer class:** v0.0.1 contained only should-refuse
  examples, which a constant-refuse model could have aced (scoring 1.0). The
  should-answer negatives in v0.1.0 (`r011`–`r014`) test that the model does
  not over-refuse questions whose answers are present in context.
- **License:** CC0 / public domain.

### `structured_extraction.jsonl` (5 records)

- **Source:** synthetic.
- **Schema fields:** `id`, `bench`, `schema_name` (`"staff_report"`),
  `document_text`, `gold_json` (5 fields: `file_number`, `applicant`,
  `location`, `recommendation`, `fiscal_impact`).
- **License:** CC0 / public domain.

### `side_by_side.jsonl` (10 records)

- **Source:** synthetic.
- **Schema fields:** `id`, `bench`, `prompt`, `rubric`.
- **License:** CC0 / public domain.

## Training corpus (planned, not yet ingested)

### Source-licensing policy (gate, not aspiration)

**No real document is ingested into `data/raw/` until its source license
clears the audit below.** The audit lives at `docs/SOURCES.md` (per-recipe)
and must be filled in before the recipe's first crawl.

For each registered jurisdiction, the recipe author records:

1. **Jurisdiction name and 2-letter state code.**
2. **Source URL pattern(s).**
3. **Site terms-of-use URL** and a verbatim quote of the section governing
   reuse, scraping, and redistribution.
4. **Robots.txt URL** and the relevant directives.
5. **Public-records statute** (e.g., California Public Records Act,
   Brown Act, Texas Public Information Act) that places the document in
   the public domain or makes it publicly inspectable.
6. **A maintainer's go/no-go decision** with one-paragraph rationale.

`san-clemente` (the reference recipe) carries this audit at
`docs/SOURCES.md#san-clemente`. New recipes that ship without the audit
filled in are rejected at PR review.

### What the policy does and does not assume

- "Publicly accessible website" ≠ "public domain." This card and
  `docs/SOURCES.md` make the per-source license explicit.
- Documents authored by U.S. state and local governments are typically not
  copyrighted (per long-standing federal principle) but **may still be
  subject to site terms** (rate limits, no-scraping clauses) and to
  state-specific records-act constraints.
- Synthetic SFT pairs derived from a source document inherit the source's
  license restrictions on **derivative works**.

### PII policy

- **Speaker labels in transcripts** are scrubbed to `[Speaker]` by default
  in `data/processed/`. Set `CIVIC_SLM_KEEP_SPEAKER_NAMES=1` to retain them
  (e.g., for elected-officials-only contexts where names are unambiguously
  public-record). The opt-out is documented in `docs/RECIPES.md`.
- **Public-comment portions** of transcripts (anything between `>> Public
Comment` and `>> End Public Comment`, or equivalent VTT cues) are
  scrubbed of speaker labels regardless of the env var, and addresses
  matching the pattern `\d+\s+[A-Z][a-z]+\s+(Street|Avenue|...)` are
  redacted to `[ADDRESS]`.
- Forks that disable scrubbing inherit the ethical and legal exposure.

### Provenance contract

Every `InstructionExample` in `data/sft/*.jsonl` records:

- `source_doc_hash` — SHA-256 of the upstream `CivicDocument` (added in
  v0.1.0).
- `source_chunk_ids` — list of `f"{doc_id}#{chunk_idx}"`.
- `Provenance.generator` — `"claude"` | `"human"` | `"model_v0"`.
- `Provenance.model` — the generator model identifier.
- `Provenance.prompt_sha` — SHA-256 of the prompt template at the time of
  generation (`src/civic_slm/synth/prompts/*.md`).
- `Provenance.created_at` — UTC timestamp.

This lineage is how we audit train/eval contamination. See next section.

### Train/eval contamination protection

- Every `EvalExample` carries an optional `source_doc_hash` (added in
  v0.1.0 to allow grandfathering current synthetic-only evals).
- `src/civic_slm/eval/runner.py` raises `ContaminationError` at startup if
  any eval example's `source_doc_hash` matches any hash in
  `data/raw/manifest.jsonl`.
- The check is opt-out (`--allow-contamination`) only with an explicit
  flag and a logged warning; it is not opt-out by default.
- The currently-shipped synthetic evals carry `source_doc_hash: null`
  (they have no upstream document); they pass the check trivially. The
  check binds the moment any real document is ingested.

## Geographic and demographic distribution (planned, will be reported when filled)

When the training corpus exists, we will publish the per-jurisdiction
distribution here, including the number of documents, the number of words,
and the share of the SFT corpus drawn from each. The v0.1.0 baseline expects
~95% San Clemente, CA — a known geographic skew called out in `MODEL_CARD.md`.

## Reproducibility

- Crawl is content-addressed by SHA-256 (`src/civic_slm/ingest/manifest.py`).
- The `data/raw/manifest.jsonl` file is committed (gitignore exception)
  even though the raw bytes are not, so the corpus is auditable without
  re-fetching.
- Synth runs are deterministic up to Anthropic API non-determinism;
  prompt-template SHAs are recorded per example.

## Versioning

- This card binds for v0.1.0. Each subsequent corpus release will append a
  `CHANGELOG`-style entry below.
- v0.1.0 (2026-04-25): refusal eval gains a should-answer negative class
  (`r011`–`r014`); `Provenance.source_doc_hash` field added; runtime
  contamination check added.
