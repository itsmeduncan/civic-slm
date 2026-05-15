# Second-city held-out eval — civic-slm-v11 (2026-05-15, closes #25)

> Auto-generated sidecar analysis. Per-jurisdiction breakdown of v1.1's
> three benches against eval examples tagged by `scripts/tag_eval_jurisdictions.py`
> (Claude Sonnet 4.6 inference; sidecar at `data/eval/.jurisdiction-tags.jsonl`,
> not corpus state — regenerable). Examples without a clear city-specific
> fingerprint are tagged `generic` and reported separately.

## Cohort definition

v1.1's SFT corpus covers 5 jurisdictions — these are the **in-corpus** set:
`san-clemente` (CA), `seattle` (WA), `boston` (MA), `denver` (CO), `cook-county` (IL).

Eval examples tagged with any other named jurisdiction are **out-of-corpus**:
`austin` (TX), `houston` (TX), `nyc` (NY), `phoenix` (AZ), `cuyahoga-county` (OH),
`atlanta` (GA), `portland-or` (OR).

The CPT corpus contains all 7 jurisdictions (cook-county + the 5 SFT + austin + atlanta
PDFs that synth failed on). For the "did v1.1 generalize to **completely** held-out
jurisdictions?" question the SFT-only definition is the strict one.

## Headline result — no second-city penalty

The #25 gate is "second-city scores within ~10% of San Clemente scores (or document the gap)."
v1.1 clears this on every bench:

| Bench      | in-corpus mean    | out-of-corpus mean | gap         | gap as % of in-corpus |
| ---------- | ----------------- | ------------------ | ----------- | --------------------- |
| factuality | **0.5185** (n=31) | **0.5073** (n=42)  | -0.0112     | -2.2%                 |
| refusal    | **1.0000** (n=16) | **1.0000** (n=18)  | 0.0000      | 0.0%                  |
| extraction | **0.4351** (n=16) | **0.4450** (n=9)   | **+0.0099** | **+2.3% (out wins)**  |

Extraction is the surprise — out-of-corpus jurisdictions score _slightly higher_ than
in-corpus. This is within noise on n=9–16, but the directional finding is clean:
v1.1 isn't memorizing in-corpus vocabulary; it's learning the underlying
"how civic documents are structured" skill.

The `generic` cohort (no city-specific fingerprint) dominates by volume on every bench
and pulls the overall mean slightly. Reported for completeness:

| Bench      | generic mean | n_generic |
| ---------- | ------------ | --------- |
| factuality | 0.4958       | 127       |
| refusal    | 0.9855       | 69        |
| extraction | 0.5926       | 25        |

## Per-jurisdiction breakdown (n ≥ 3)

### factuality

| Jurisdiction      | cohort        | n   | mean   |
| ----------------- | ------------- | --- | ------ |
| `austin`          | out-of-corpus | 7   | 0.5932 |
| `seattle`         | in-corpus     | 5   | 0.5702 |
| `boston`          | in-corpus     | 5   | 0.5524 |
| `portland-or`     | out-of-corpus | 11  | 0.5229 |
| `nyc`             | out-of-corpus | 9   | 0.5095 |
| `san-clemente`    | in-corpus     | 17  | 0.5030 |
| `cuyahoga-county` | out-of-corpus | 7   | 0.4493 |
| `houston`         | out-of-corpus | 6   | 0.4398 |

Austin (out-of-corpus, Texas SUP/TIRZ vocabulary) is the highest-scoring jurisdiction
on factuality — striking, given Austin docs weren't in the SFT corpus.

### refusal

| Jurisdiction   | cohort        | n   | mean   |
| -------------- | ------------- | --- | ------ |
| `san-clemente` | in-corpus     | 9   | 1.0000 |
| `austin`       | out-of-corpus | 6   | 1.0000 |
| `nyc`          | out-of-corpus | 3   | 1.0000 |
| `portland-or`  | out-of-corpus | 3   | 1.0000 |

Refusal is uninformative at this slicing — saturated at 1.0 across all tagged groups.

### extraction

| Jurisdiction   | cohort        | n   | mean   |
| -------------- | ------------- | --- | ------ |
| `austin`       | out-of-corpus | 3   | 0.4411 |
| `san-clemente` | in-corpus     | 10  | 0.3848 |

Most extraction jurisdictions had n<3 after tagging. The two that did:
austin (out) beats san-clemente (in) on n=3 vs n=10 — same generalization signal as factuality.

## Methodology caveats

1. **Tagger noise.** ~60–85% of examples got `generic` (no city-specific fingerprint).
   Some of these are genuinely jurisdiction-agnostic; some are sc-styled but lack
   the surface markers Claude needed to pin them. A more aggressive tagger (e.g.
   one that infers from staff-report templates) could shift the split.
2. **Sample size.** Per-juris n is small (3–17 outside generic). Differences within
   a single bench should be read as directional, not significant.
3. **Tags are a sidecar, not corpus state.** A v0.3.x schema change would add
   `jurisdiction: str | None` to `_EvalBase` and let `civic-slm eval run` natively
   emit per-juris breakdowns. Until then, this analysis is regenerable via
   `scripts/tag_eval_jurisdictions.py`.

## Reproduction

```bash
# Tag eval examples (one-time, ~$2 via Claude Sonnet 4.6).
uv run python scripts/tag_eval_jurisdictions.py

# Recompute breakdown (reads tags + artifacts/evals/civic-slm-v11/*.json).
# Inline in this commit; will be wired into a `civic-slm eval slice` CLI in v0.3.x.
```

## Decision

**v1.1 clears the #25 gate on all 3 measured benches.** The extraction story plus
the no-second-city-penalty story together justify treating v1.1 as the
candidate v0.3.0 release. Open work for the release:

- `side_by_side` bench (not yet run on v1.1)
- base re-baseline under the new `--no-thinking max_tokens=1024` defaults so the
  base/v1/v1.1 columns are all apples-to-apples (currently the v1.1 ↔ v1 comparison
  is clean; v1.1 ↔ base still inherits the older base measurement protocol)
- the `civic-slm eval slice --by jurisdiction` proper API (replace this sidecar)
