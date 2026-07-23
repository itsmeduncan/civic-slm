# Model Card — civic-slm-e4b-v1

**civic-slm-e4b-v1** is a LoRA fine-tune of **Google Gemma 4 E4B** (the
effective-4B MatFormer / Per-Layer-Embedding variant, `gemma3n` arch in
mlx-lm), specialized for U.S. local-government document understanding and
built to run **on-device** — a laptop or a phone, not a datacenter. It is
distributed as **MLX 4-bit** (primary) and **GGUF Q5_K_M** (llama.cpp /
Ollama).

The design bar is **edge deployability**, so the model is measured against
**its own base (Gemma 4 E4B)** — the must-beat floor — not against a
27B/72B accuracy ceiling. A modest accuracy gain that ships on-device beats a
larger model that doesn't. The `side_by_side` bench keeps a larger model
(Gemma 4 31B) in view only as a _reference point_ for how the edge model fares
against a model ~7× its size, never as a target to match.

> **v1 scope: CPT + SFT only.** The E4B v1 recipe is continued-pretraining
> followed by supervised fine-tuning, merged and quantized. **DPO is
> deferred** to a later release — the preference-optimization stage is not part
> of v1. The training pipeline still defines a DPO stage (`configs/dpo.yaml`),
> but no DPO adapter is included in the v1 weights.

## Evaluation of record

_Measured 2026-07-23._ All four benches were run **apples-to-apples** — candidate and base under
identical flags: `seed=0`, `temperature=0.0`, `--max-tokens 1024`,
`--no-thinking`, word-overlap similarity, and `--drop-contaminated`
(train/eval-overlapping examples dropped from both columns by the same rule, so
`n` and `contamination_dropped` match per bench). Raw run configs and
per-example JSON live at `artifacts/evals/civic-slm-e4b-v1/` (candidate) and
`artifacts/evals/base-gemma-4-e4b/` (base).

| Bench          | n   | Gemma 4 E4B (base) | **civic-slm-e4b-v1** | Δ vs base          | v1 target       | Result                                   |
| -------------- | --- | ------------------ | -------------------- | ------------------ | --------------- | ---------------------------------------- |
| `factuality`   | 196 | 0.460              | **0.561**            | **+0.101 (+22%)**  | ≥ 0.65 (aspir.) | ✅ beats base; below aspirational target |
| `refusal`      | 100 | 0.990              | **0.970**            | −0.020             | maintain ≥ 0.95 | ✅ held above the 0.95 floor             |
| `extraction`   | 44  | 0.097              | **0.682**            | **+0.585 (+603%)** | ≥ 0.60          | ✅ clears target decisively              |
| `side_by_side` | 100 | — (not run)        | **0.115**            | n/a                | reference only  | ℹ️ 18% non-loss (5W/13T/82L) vs 31B ref  |

**Gate — meets its v1 target on 3/3 scoreable benches.** The three scoreable
benches are a clean apples-to-apples base-vs-candidate comparison: the fine-tune
**beats the base outright on factuality (+22%) and extraction (+603%)** and
**holds refusal above its 0.95 floor** (a −0.02 move within single-example noise
on n=100 — a maintain-target, not a beat). Extraction is the headline: the base
E4B has essentially no structured-extraction ability (0.097) across the seven
schemas, and the fine-tune lifts it to 0.682.

`side_by_side` is **not** part of the beat-base gate — it is candidate-vs-`Gemma
4 31B` (`comparator-gemma-4-31b` → `gemma-4-31B-it-MLX-8bit`, MLX 8-bit, served
locally), a reference read on how a 4B edge model fares against one ~7× larger.
The candidate wins or ties **18/100** head-to-heads (5 wins, 13 ties, 82 losses;
win-rate 0.115). A base-vs-31B run was **not** performed, so this is a raw
standing against the ceiling, not a quantified gap-closure vs the base.

### Recipe

- **CPT:** LoRA continued-pretraining, 2000 iters, on the multi-jurisdiction
  civic corpus. Output `artifacts/gemma-e4b-civic-cpt/`.
- **SFT:** LoRA instruction tuning, **1 epoch, LR 1e-4**. Output
  `artifacts/gemma-e4b-civic-sft/`.
- **Merge + quantize:** adapters fused into the base, exported MLX 4-bit
  (`artifacts/civic-e4b-v1-mlx-q4`, registry label `civic-slm-e4b-v1`) and
  GGUF Q5_K_M.
- Hyperparameters in `configs/gemma-e4b-{cpt,sft}.yaml`; design rationale in
  `ARCHITECTURE.md`.

**Why these knobs.** An epoch sweep (1/2/3) showed **1 epoch is optimal** — at
3 epochs the model overfit and extraction _collapsed_ (0.757 → 0.270) with
non-schema field-name drift. **LR 1e-4 (vs 2e-4) was required to preserve
refusal**: at 2e-4 the answer-heavy SFT corpus eroded refusal to ~0.83, below
the floor. The shipped recipe is the point on the sweep that clears extraction
while keeping refusal safe.

## Model details

- **Name:** civic-slm-e4b-v1
- **Base model:** `google/gemma-4-e4b` (Gemma 4 E4B — effective-4B
  MatFormer / Per-Layer-Embedding variant, `gemma3n` arch in mlx-lm), MLX 4-bit
  as published in LM Studio's catalog (`lmstudio-community/gemma-4-E4B-it-MLX-4bit`).
  Chosen for on-device deployment. _Previous bases (retired at the 2026-07-21
  edge-first pivot):_ Qwen 3.6 27B (`qwen3.6-27b-ud-mlx`), Qwen 2.5 7B — see the
  [prior (pre-pivot) results](#prior-pre-pivot-results-qwen-era) below.
- **Adaptation method:** LoRA continued-pretraining + LoRA SFT (**CPT → SFT, no
  DPO in v1**), merged and quantized.
- **Released formats:** MLX 4-bit (primary), GGUF Q5_K_M (llama.cpp / Ollama).
- **Code license:** MIT (see `LICENSE`).
- **Weights license:** _To be decided before HF-Hub publish._ Candidates:
  Apache-2.0 or OpenRAIL-M with the use restrictions in
  `ACCEPTABLE_USE_POLICY.md`.
- **Dataset license:** _Per-source._ See `DATA_CARD.md` — the training corpus
  inherits whatever license each crawled civic document is published under, and
  the synthetic SFT pairs are derivative works of those source documents.
- **Maintainers:** see `README.md`.
- **Contact for issues:** `itsmeduncan@gmail.com`.

### Base-model integrity

The base model `google/gemma-4-e4b` is downloaded through LM Studio's model
catalog (which fetches from Hugging Face under the hood). To prevent a silent
upstream re-quantize or weight-tampering incident from moving the eval floor
under us, the training configs (`configs/{cpt,sft,dpo}.yaml`) accept an optional
`base_model_revision` field — a branch name, a tag, or a 40-char git commit SHA.
Recommended posture before any release:

1. Note the LM Studio model download date and the corresponding HF revision SHA
   (LM Studio's model-details panel shows the resolved repo + revision).
2. Pin `base_model_revision` to that 40-char SHA in the config.
3. Re-run the four eval benchmarks against the pinned revision and commit the
   baselines under `artifacts/evals/base-gemma-4-e4b/`.
4. Update the pin only when re-running and re-committing the baselines —
   otherwise prior numbers stop being comparable.

Strict-local mode (`CIVIC_SLM_STRICT_LOCAL=1`) does **not** prevent HF model
downloads (documented in `docs/RUNTIMES.md`). The revision pin is the integrity
story for HF; the strict-local tripwire is the integrity story for paid-API
spend.

## Intended use

Helping civic technologists, journalists, public servants, and residents
understand publicly-available U.S. local-government documents — agendas, staff
reports, meeting minutes, ordinances, and municipal codes.

See `ACCEPTABLE_USE_POLICY.md` for the full list of intended uses and prohibited
uses (including: legal advice, voter-eligibility determinations,
benefits-eligibility determinations, surveillance of named individuals).

### Out of scope

- Anything in `ACCEPTABLE_USE_POLICY.md`.
- Non-U.S. jurisdictions.
- Tribal-government or federal documents.
- Real-time or post-cutoff information; the model has a hard training-cutoff
  date.

## Training data

See `DATA_CARD.md` for full details. Summary:

- **Continued-pretraining corpus:** raw text from crawled U.S. local-government
  documents and meeting transcripts.
- **Supervised fine-tuning corpus:** synthetic instruction pairs generated by
  Claude Opus 4.7 conditioned on chunks from the CPT corpus, validated against a
  Pydantic schema, with the first 500 examples human-reviewed.
- **DPO corpus:** preference pairs over civic-task outputs — **defined but not
  used in v1** (DPO is deferred; see the scope note at the top of this card).
- **Provenance:** every SFT example records the source-document SHA-256, the
  prompt-template SHA-256, the generator model, and a UTC timestamp
  (`src/civic_slm/schema.py` `Provenance`).

## Evaluation methodology

Four held-out benchmarks live in `data/eval/`; the harness is in
`src/civic_slm/eval/`. To reproduce a scoreable bench:

```
civic-slm eval run --model civic-slm-e4b-v1 --bench <name> \
  --bench-file data/eval/<file>.jsonl \
  --seed 0 --temperature 0 --max-tokens 1024 --no-thinking \
  --similarity word_overlap --drop-contaminated
```

The `side_by_side` bench is pairwise, judged by Claude Sonnet with A/B
position-swap (a model only wins if it wins both orderings, else tie):

```
civic-slm eval side-by-side --candidate civic-slm-e4b-v1 \
  --comparator comparator-gemma-4-31b
```

| Benchmark               | n   | What it measures                                                    |
| ----------------------- | --- | ------------------------------------------------------------------- |
| `civic_factuality`      | 196 | citation exact-match + answer similarity (word-overlap or BGE)      |
| `refusal`               | 100 | refusal recall + over-refusal precision (mixed positives/negatives) |
| `structured_extraction` | 44  | field-level F1 vs. gold JSON across seven schemas                   |
| `side_by_side`          | 100 | LLM-judged pairwise win-rate vs. `gemma-4-31B-it-MLX` (reference)   |

(`n` is post-contamination-drop; the raw benches are 200/103/50/100. Extraction
schemas: `staff_report`, `meeting_metadata`, `meeting_agenda_item`, `ordinance`,
`resolution`, `public_hearing_notice`, `contract_award`.)

### Serving for eval

Both the candidate (MLX 4-bit) and the 31B reference comparator are served
locally via `mlx_lm.server` with `--chat-template-args '{"enable_thinking":
false}'` so the comparison is thinking-off on both sides. This matters for
`side_by_side`: the eval client reads only visible `content` (not any hidden
`reasoning` field), so a comparator left in thinking mode would spend its token
budget on chain-of-thought and be unfairly truncated. See `docs/RUNTIMES.md`.

### Known limitations of the eval harness (be honest)

- **Word-overlap factuality scorer (default).** The default factuality scorer is
  word-overlap (Jaccard over token sets), which rewards verbatim copying and
  penalizes correct paraphrase. An opt-in BGE dual-encoder scorer is available
  via `--similarity bge` (`BAAI/bge-large-en-v1.5`). Numbers under the two
  scorers are **not** comparable; the run-config header records the choice.
- **Regex refusal detector.** The refusal scorer matches a small set of English
  refusal patterns (`src/civic_slm/eval/scorers.py`). It is brittle to wordings
  outside that set.
- **Single seed.** Reported numbers are a single seed at temperature 0; seed and
  temperature are logged in every eval JSON.
- **California-leaning eval data.** Original examples are San-Clemente-styled;
  the bench now spans ~30 U.S. jurisdictions (SUP, TIRZ, ULURP/SEQRA, CDBG,
  LIHTC, home-rule vs. Dillon's Rule), but coverage is uneven.
- **`side_by_side` sampling.** The pairwise bench uses the harness default
  generation settings (512-token cap, applied equally to both models) rather
  than the 1024-cap used for the scoreable benches; both models get the same
  budget, so the comparison stays symmetric.

## Bias, risks, and known failure modes

- **Geographic bias.** Training data is weighted toward California municipal
  documents, with San Clemente, CA as the demo jurisdiction. Expect worse
  performance on Texas, New York, or Ohio documents until more recipes ship. Do
  **not** present this as a universal U.S. model.
- **Vocabulary bias.** The model preferentially uses California-flavored
  terminology (CEQA, CUP, ABAG/SCAG vocabulary) even for jurisdictions where
  those terms don't apply.
- **Public-comment PII.** Public-comment portions of transcripts may contain
  residents' names, addresses, and personal stories heard by a local audience,
  not intended for a globally-distributed LLM. An opt-in
  `--scrub-public-comment` flag defaults to scrubbing in `data/processed/`.
  Forks that disable scrubbing inherit the ethical and legal exposure.
- **Hallucination on missing context.** Despite refusal training, the model can
  still fabricate citations or invent ordinance numbers when pushed. Always link
  end-users to the cited source document so they can verify.
- **Adversarial prompt injection.** Civic documents containing injection-style
  instructions may influence the SFT corpus. Mitigations are human review of the
  first 500 SFT examples and Pydantic schema validation; no adversarial-prompt
  detector is in the loop.

## Environmental impact

Training is single-Mac, Apple-Silicon, LoRA-only. The E4B v1 recipe is CPT + SFT
(no DPO), so wall-clock is a few hours on M-series unified memory. Carbon impact
is small relative to a from-scratch pretrain. The synth-data step issues Claude
Opus 4.7 calls; the upstream provider's footprint is not accounted for here.

## Caveats and recommendations

- Always show end-users the citation and let them open the source document.
- Do not use this model unsupervised in any user-facing flow that affects
  benefits, eligibility, or legal status. See `ACCEPTABLE_USE_POLICY.md`.
- Re-evaluate before deploying to a jurisdiction not in the training data — at
  minimum, run the four benchmarks against held-out documents from that
  jurisdiction.

## Versioning and release status

- **civic-slm-e4b-v1** (this card) — Gemma 4 E4B, CPT + SFT, merged MLX 4-bit at
  `artifacts/civic-e4b-v1-mlx-q4`. Eval-of-record at
  `artifacts/evals/civic-slm-e4b-v1/` (candidate) and
  `artifacts/evals/base-gemma-4-e4b/` (base).
- **HF-Hub publish:** pending the maintainer publish decision, a weights-license
  choice, and San Clemente source-license finalization (`docs/SOURCES.md`). Once
  weights ship, a per-version receipt lands at
  `artifacts/civic-e4b-v1/MODEL_CARD.md` (this top-level card is the contract;
  the per-version one is the receipt).
- **Deferred to a later release:** DPO (preference optimization) and lifting
  factuality to the ≥ 0.65 aspirational target.

## Prior (pre-pivot) results (Qwen era)

> These numbers are **retained for provenance only**. They were measured before
> the 2026-07-21 edge-first pivot, against **different base models** (Qwen 2.5
> 7B, then Qwen 3.6 27B), often under **non-apples-to-apples flags** (reasoning
> on, `max_tokens=4096`). **They do not define the v1 gate** and are **not
> comparable** to the E4B numbers above. Raw JSONs remain under
> `artifacts/evals/base-qwen*/`, `artifacts/evals/san-clemente-v1/`, and
> `artifacts/evals/civic-slm-v11/`.

Two Qwen 3.6 27B fine-tunes were measured (bench sizes 200/103/50/100):

| Benchmark               | Base Qwen 3.6 27B | v1 (san-clemente-v1) | v1.1 (civic-slm-v11) |
| ----------------------- | ----------------- | -------------------- | -------------------- |
| `civic_factuality`      | 0.4952            | 0.5025               | 0.5017               |
| `refusal`               | 1.000             | 0.9903               | 0.9903               |
| `structured_extraction` | 0.2735            | 0.1406               | 0.5157               |
| `side_by_side`          | n/a               | not run              | not run              |

Notes retained for the record: the Qwen base column was measured with reasoning
**on** and `max_tokens=4096` while the fine-tune columns used reasoning **off**
and `max_tokens=1024`, so the Qwen base-vs-fine-tune gap was never a clean
head-to-head — one of the motivations for re-baselining cleanly on E4B. The
v1.1 multi-jurisdiction retrain (5 jurisdictions, 3,002 SFT examples) cleared a
second-city held-out generalization check (#25); its extraction jump
(0.2735 → 0.5157) was the strongest pre-pivot signal. The prior base was Qwen
2.5 7B before Qwen 3.6 27B. None of this carries into the E4B v1 gate.
