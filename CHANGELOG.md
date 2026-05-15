# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — v0.2.x v1.1 multi-jurisdiction fine-tune (`civic-slm-v11`)

Second v1 fine-tune, trained on the multi-jurisdiction corpus from the previous changelog entry. Headline result: **structured-extraction 0.14 → 0.52** (the regression v1 introduced is gone, and v1.1 is now well above the base's 0.27 on that bench).

- **Training:** 200-iter CPT on 7-juris corpus (495 chunks) + 3-epoch SFT on 5-juris corpus (3,002 examples; 2,702 train / 300 valid). Wall-clock ~14.5 hr on M-series 128GB unified memory. Pipeline: `configs/multi.cpt.yaml` + `configs/multi.sft.yaml`, kicked off by an ad-hoc script (will be wired into `civic-slm train multi-jurisdiction <slug1> <slug2>...` in v0.3.x — see follow-up issue).
- **Eval (max_tokens=1024, --no-thinking, n=200/103/50):**
  | Bench | base | v1 (sc) | **v1.1 (multi)** |
  |---|---|---|---|
  | factuality | 0.4952 | 0.5025 | **0.5017** (flat) |
  | refusal | 1.000 | 0.9903 | **0.9903** (same as v1) |
  | extraction | 0.2735 | 0.1406 | **0.5157** (+88% vs base, +267% vs v1) |
- **Gate status:** still doesn't clear the strict 3/4-bench rule (factuality + refusal are flat-to-noise vs base; side_by_side not run). Extraction win is decisive though — the corpus-size hypothesis from v1's gap analysis is confirmed. v1.1 is the **candidate v0.3.0 release**.
- **Model registry:** new `civic-slm-v11` entry in `src/civic_slm/serve/models.py` pointing at `artifacts/multi-v11-mlx-q4`.
- **Raw evals:** `artifacts/evals/civic-slm-v11/{factuality,refusal,extraction}.{json,md}`.
- **Configs committed:** `configs/multi.cpt.yaml` + `configs/multi.sft.yaml` so the run is reproducible against a future maintainer's corpus.
- **Total project spend at this milestone:** ~\$86 on Anthropic synth (no Anthropic calls during training/eval; those are 100% local MLX).

### Added — v0.2.x multi-jurisdiction corpus + synth model env override

First real multi-jurisdiction crawl + synth run, foundation for the v0.3.x retrain.

- **Corpus grew from 28 docs → 61 docs / 1 jurisdiction → 7 jurisdictions.** Crawled the 8 GO'd Legistar recipes (5 new docs each); seattle/boston/denver/cook-county/atlanta/austin landed. NYC and portland-or did **not** — see below.
- **Synth corpus: 414 → 3,002 examples (7.3×).** Generated on Claude Opus 4.7 across san-clemente (414, existing) + seattle (108) + boston (240) + denver (323) + cook-county (1,917). atlanta attempted on Sonnet 4.6 and failed mid-run (Anthropic 529 overloaded errors crashed the run before write); austin skipped.
- **`CIVIC_SLM_SYNTH_MODEL` env override.** New env var on the synth backend lets cost-sensitive runs drop from Opus to Sonnet 4.6 (~3× cheaper) or Haiku 4.5 (~10× cheaper) without code changes. Per-call cost was the main miss in the v0.2 plan ($66 spent before we paused to course-correct).
- **Bugfix: `granicus_legistar.md` vendor template had a stray `{jurisdiction}` placeholder** in the Legistar REST-API documentation note that crashed `str.format(...)` at crawl time with `KeyError`. Escaped to `{{jurisdiction}}` so format() emits it literally; new regression test in `tests/test_recipes.py` covers the round-trip.
- **`DATA_CARD.md`** auto-refreshed via `civic-slm data-card --write`: 61 docs, 495 chunks, 410k tokens across 7 jurisdictions.
- **Crawl misses tracked separately** as #59 (portland-or — not actually a Legistar tenant; needs `instruction:` override). NYC's Legistar-with-58-GUIDs page consistently times out browser-use; will track in a follow-up issue if a custom instruction doesn't recover it on next pass.
- **Cost note for posterity:** Opus 4.7 at ~$0.075/call on this workload, _not_ the $0.02/call the v0.2 plan assumed. Future synth at scale should default to Sonnet 4.6 via `CIVIC_SLM_SYNTH_MODEL` unless quality testing justifies Opus.

### Changed — v0.2.x SOURCES audit cohort flip to GO

- **`docs/SOURCES.md`.** All 8 v0.2 Legistar jurisdictions (`seattle`, `nyc`, `boston`, `denver`, `portland-or`, `cook-county`, `atlanta`, `austin`) flipped from `Decision: PENDING` to `Decision: GO` under a documented **blanket maintainer posture** (`docs/SOURCES.md` § "Maintainer GO posture — v0.2.x Legistar cohort"). The posture is intentionally narrower than san-clemente's per-site audit: per-jurisdiction ToS verbatim quotes and robots.txt are deferred to a v1.1+ backfill; the fair-use stance and right-of-withdrawal are inherited from san-clemente. Reversible per-jurisdiction — if a specific site's ToS turns out to forbid this use, that entry flips back to NO-GO and its corpus is removed from `data/raw/`/`processed/`/`sft/` within 30 days. This unblocks the multi-jurisdiction crawl + synth scale-up tracked in #21. santa-monica stays PENDING (different vendor — IQM2 — outside the Legistar cohort).
- **Filed #57** ("v1.1+: generative curation pass over the synth corpus"). Spec for replacing or layering on top of the manual `review-sft` loop with a model-driven curator that classifies systemic defects, stack-ranks by likely badness, and triages into auto-accept / human-review / reject tiers. Not a v1 blocker — v1 ships on hand-curated 500 + uncurated tail.

### Added — v0.2.x auto-generated per-jurisdiction data card (closes #26)

- **`civic-slm data-card`.** Scans `data/raw/manifest.jsonl` plus `data/processed/{slug}.jsonl`, groups by jurisdiction, and emits a markdown table: doc count, chunk count, token count, doc-type distribution, crawl date range. Three modes: stdout (default, no flags), `--write` splices into `DATA_CARD.md` between `<!-- DATA_CARD:JURISDICTIONS:BEGIN -->` / `END` sentinels, `--check` exits non-zero if the on-disk table would change (designed as a CI gate per #55). Splice is idempotent — second `--write` produces a byte-identical file.
- **`DATA_CARD.md`** now has the sentinel block + the first auto-generated breakdown (san-clemente, CA, 28 docs / 35 chunks / 24,506 tokens).
- **Tests:** `tests/test_data_card.py` covers empty manifest, single-jurisdiction roll-up, sort-stability, sentinel-missing error, idempotent splice (regression-tested — the first cut added a blank line per write).

### Added — v0.2.x first measured v1 fine-tune

- **`artifacts/evals/san-clemente-v1/`.** First v1 fine-tune trained and measured end-to-end via the per-jurisdiction pipeline (PR #49). 29 docs → 35 chunks → 414 SFT examples; CPT val 2.48 → 0.82, SFT val 2.37 → 0.58. Eval scores at `max_tokens=1024, --no-thinking`: factuality **0.5025**, refusal **0.9903**, extraction **0.1406** (n=200/103/50).
- **`MODEL_CARD.md` flips from planned → measured.** New "v1 (san-clemente-v1)" column alongside base; honest gap analysis explaining that v1 does **not** clear the eval gate (≥ 3/4 benches beating base). Root cause documented: 414-example corpus is an order of magnitude too small for the 200/100/50/100 multi-jurisdiction bench. Path forward (corpus scale-up via #21 + retrain) called out explicitly. Asymmetric measurement caveat documented: base column was measured under reasoning-on / max_tokens=4096; v1 column under reasoning-off / max_tokens=1024 — base re-baseline under the new defaults is a prerequisite for a clean head-to-head.
- **`README.md`.** Status line flips from "all code-only tracks landed" → "first v1 trained and measured locally, has not cleared the gate." Stale bench counts (25/29/15/25) corrected to actual (200/103/50/100).
- **`.gitignore`.** Per-slug pipeline outputs (`artifacts/{slug}-cpt/`, `-sft/`, `-v1-fused/`, `-pipeline/`) ignored so weights stay local; `artifacts/evals/` still tracked so MODEL_CARD numbers reproduce.

### Added — v0.2.x multi-jurisdiction recipes (closes #24)

- **Eight new YAML recipes:** `seattle`, `nyc`, `boston`, `denver`, `portland-or`, `cook-county`, `atlanta`, `austin`. All Legistar tenants — vendor template `granicus_legistar.md` covers them — geographically distributed across Pacific NW, Mountain West, Midwest, Northeast, South, and Texas, with `cook-county` as the first county-level recipe (vs. city). The "all 50 states" framing in the README is no longer aspirational on the corpus-diversity axis; it remains aspirational on the audit axis (every recipe ships **Decision: PENDING** in `docs/SOURCES.md` — maintainer audits each before crawl).
- **`docs/SOURCES.md`** gains a PENDING audit stub for each new jurisdiction with the relevant state public-records statute pre-filled, awaiting per-site ToS / robots.txt capture before flip to GO.
- **`docs/RECIPES.md` "Currently registered"** table lists all 10 shipped recipes with vendor + jurisdiction type + region so a contributor sees the gap before adding another Legistar city. Platform coverage callout makes the next contribution opportunity explicit: **Municode and PrimeGov are not yet represented.**

### Added — v0.2.x eval scale-up tooling

- **`civic-slm eval seed <jurisdiction> --bench {factuality,refusal,extraction,side_by_side}`.** Drafts eval-bench candidates from real civic chunks via the configured LLM backend (LM Studio by default). All four benches now wired with per-bench prompt templates and validators:
  - **factuality** — hard-rejects candidates whose `gold_citations` aren't verbatim substrings of the source chunk (contamination guard).
  - **refusal** — locks `context` to the chunk verbatim so the model can't paraphrase its way past the bench; expected_refusal defaults to true.
  - **extraction** — requires a flat `gold_json` (nested objects rejected; current scorer is flat-F1) and a snake_case `schema_name`.
  - **side_by_side** — emits open-ended prompt + concrete rubric for the pairwise judge.
    Stages to `data/eval/.staged-{bench}.jsonl` so the maintainer reviews before promoting into the canonical bench file. `--promote` skips staging once a batch is trusted. Built for #16's "200 / 100 / 50 / 100" target.

### Added — v0.2.x developer playground + ingest/synth CLIs

- **`civic-slm process <jurisdiction>`.** Reads manifest entries for a jurisdiction, extracts text from each PDF under `data/raw/`, chunks with the existing 1024/128 chunker, and writes `data/processed/{jurisdiction}.jsonl`. Replaces the previous "chunk lazily inside synth" inline-Python recipe in `docs/USAGE.md`.
- **`civic-slm synth <jurisdiction>`.** Typer wrapper around `civic_slm.synth.generate.generate_corpus()`. Loads processed chunks, resolves the 2-letter state and dominant `doc_type` from the manifest, runs the async corpus generator under `asyncio.run`, and writes `data/sft/{jurisdiction}.jsonl`. Resume is on by default; `--no-resume` forces a full re-run. Backend selection still goes through `CIVIC_SLM_LLM_BACKEND` — see `docs/RUNTIMES.md`.
- **`web/` — Next.js + assistant-ui chat playground.** Dogfooding UI for the candidate model. Streaming `useLocalRuntime` + `ChatModelAdapter` → `/api/chat` (an OpenAI-shape proxy that defaults to `CIVIC_SLM_CANDIDATE_URL`). Four task presets (general, extraction, fact-check, summarize) swap system prompts without leaving the thread. Model dropdown defaults to **Gemma 4 (local)** and exposes the trained Civic SLM slot and base Qwen 2.5 for side-by-side prompt sniffing. Per-slot model strings are overridable via `CIVIC_SLM_GEMMA_MODEL`, `CIVIC_SLM_CIVIC_MODEL`, `CIVIC_SLM_CANDIDATE_MODEL`. Run with `pnpm --dir web dev`.

### Added — v0.2.x Track A3: 72B comparator wiring + smoke

- **`side_by_side` fails fast on a missing comparator.** A 100-example bench used to crash on the first chat call after warming up the candidate; now it pings `$CIVIC_SLM_TEACHER_URL` before doing anything else and raises `ComparatorMissingError` with a pointer to `docs/RUNTIMES.md` if the teacher isn't up. No candidate-side tokens get burned on a misconfigured run.
- **`civic-slm doctor --teacher`.** New flag forces the teacher-URL ping even when `CIVIC_SLM_LLM_BACKEND` isn't `local`. Use this before starting a side_by_side eval to confirm the 72B comparator is reachable.
- **`docs/RUNTIMES.md` — "Standing up the 72B comparator"** section. Copy-paste recipe for downloading `Qwen2.5-72B-Instruct-Q4_K_M.gguf` (~40GB), running `llama-server` on port 8081 alongside the candidate on 8080, pointing civic-slm at it, and verifying with `civic-slm doctor --teacher`. Includes hardware reality-check (disk, RAM, throughput) and a 32B fallback for under-spec'd Macs.
- **Tests:** `tests/test_side_by_side.py` covers win-rate computation against a stubbed judge (always-A and always-tie), and the comparator-missing error path against an unallocated localhost port.

### Added — v0.2.x Track A1: BGE-reranker factuality scorer

- **`civic-slm eval run --similarity {word_overlap,bge}`.** New flag. Default stays `word_overlap` so pre-v0.2 baselines remain bit-reproducible; opt into `bge` to get BAAI/bge-large-en-v1.5 dual-encoder cosine, mapped to `[0, 1]`. The choice is recorded in the eval JSONL `_run_config` header alongside `bge_model`.
- **`civic_slm.eval.embeddings.bge_similarity_fn(model_id=...)`** — lazy-loaded helper, caches the encoder in module state. The encoder is only imported when `--similarity bge` is selected, so the default install does not pull `sentence-transformers` into the import graph.
- **`civic-slm eval run` records `similarity` and `bge_model`** in the run config header, so a markdown report immediately tells you which scorer produced the numbers.
- **Tests:** `tests/test_embeddings.py` covers the `_resolve_similarity` mapping and an empty-input short-circuit; the actual model-download check is gated behind `CIVIC_SLM_RUN_BGE_TEST=1` so the default `pytest` stays fast and offline.

### Breaking — eval

- **Factuality numbers under `--similarity bge` are not comparable to numbers under `--similarity word_overlap`.** They use different scales. Pre-v0.2 baselines in `artifacts/evals/base-qwen2.5-7b/factuality.{json,md}` were produced under word-overlap and remain valid for that scorer. v0.2 baselines under BGE will be added in a separate file (`factuality.bge.{json,md}`) when the maintainer re-runs them; until then, **do not mix the two**. Reports now record `similarity:` so this is auditable per-run.

### Added — v0.2.x Track A2: training pipeline robustness

- **Subprocess supervisor** (`src/civic_slm/train/supervisor.py`). All three trainer wrappers (`cpt.py`, `sft.py`, `dpo.py`) now run `mlx_lm` through `run_supervised(cmd)`, which propagates `SIGTERM` and `SIGINT` to the child so a Ctrl-C lets `mlx_lm` flush a checkpoint cleanly. After a 10s grace, an unresponsive child is escalated to `SIGKILL`. Non-zero exits raise `TrainerError` with the exit code in the message.
- **Resume guard.** `civic-slm train cpt|sft|dpo` now refuses to start if the configured `output_dir` already contains an adapter (`*.safetensors`). Pass `--resume` to continue training from the existing adapter, or move/delete the directory to start fresh. Previous behavior silently overwrote the prior run.
- **`--smoke-test` flag** on `cpt`, `sft`, and `dpo`. CPT runs 100 iters, SFT/DPO 50 steps. Skips the resume guard since smoke runs are throwaway. Per the CLAUDE.md working agreement: "before running long training jobs, do a dry-run at 100 steps."
- **Tests:** `tests/test_supervisor.py` covers happy-path zero exit, non-zero raise, signal-forwarding via a mocked `Popen`, and the resume-guard `has_existing_adapter` detector. Cross-process SIGINT propagation is verified by a smoke recipe in `RELEASING.md` rather than CI (too flaky to be load-bearing in pytest).

### Added — v0.2.x Track C3 (partial): eval scale-up + multi-jurisdiction seeding

The four eval benches grow from 39 total examples (10/14/5/10) to 94 (25/29/15/25). Every new example draws from a non-California jurisdiction so the v1 eval harness has a defensible "second city" signal before any training claim is published.

- **`data/eval/civic_factuality.jsonl`** — 10 → 25. New examples cover Austin TX, Houston TX, Cuyahoga County OH, NYC, Phoenix AZ, Seattle WA, Cook County IL, Atlanta GA, Boston MA, Denver CO, Portland OR. Vocabulary that doesn't exist in the v0 set: SUP (vs. CUP), TIRZ, FAR, CDBG, LIHTC, CEQR (vs. CEQA), home-rule, fiscal-note, supplemental appropriation.
- **`data/eval/refusal.jsonl`** — 14 → 29. The new 15 examples maintain the should-refuse / should-answer balance: 8 should-refuse against multi-jurisdiction context (where the answer is genuinely missing) and 7 should-answer (where the answer is squarely in the cited context). The over-refusal precision signal is now stronger.
- **`data/eval/structured_extraction.jsonl`** — 5 → 15. Multi-jurisdiction `staff_report` schema examples covering rezonings, contract authorizations, supplemental appropriations, brownfield remediation, and ZBA cases — the field shapes vary across jurisdictions in ways the v0 set didn't capture.
- **`data/eval/side_by_side.jsonl`** — 10 → 25. Prompts now exercise: SUP vs. CUP, TIRZ, ULURP/SEQRA, Ohio resolution structure, brownfield funds, consent agendas, ordinance-vs-resolution, home-rule vs. Dillon's Rule, CIP, public-records timelines (with explicit acknowledgement of variation), CDBG flow, fiscal-note contents, pre-emption, Texas general-law vs. home-rule cities, LIHTC mechanics.
- **Test update:** `test_load_factuality_examples_validates` and `test_runner_round_trip` now assert load+validate behavior, not the exact count, so the bench can grow in future PRs without test churn.

### Notes — eval scale-up

Targets per `ROADMAP.md` v0.2.x are 200/100/50/100; this PR is roughly 50% of refusal, 30% of extraction, 25% of side_by_side, and 12% of factuality. Further authoring + the synthetic-source-document path (real crawl → real chunks → eval examples bound to real `source_doc_hash`es) lands in subsequent commits and exercises the contamination check at `civic_slm.eval.runner.assert_no_contamination()`.

### Added — remaining MEDIUM/LOW tier from `AUDIT.md`

- **Synth prompt-injection mitigation.** Prompt templates now wrap chunk text in `<civic_document>...</civic_document>` tags and instruct the generator to treat the tagged region as data, not instructions. `synth.generate._safe_chunk_text()` redacts any literal `</civic_document>` inside source text to `[redacted-close-tag]` so a hostile civic document can't break out of the data section. Closing-tag matches are logged. (Audit §3 MEDIUM.)
- **HF model-weight integrity posture.** `TrainConfig` accepts an optional `base_model_revision` (branch, tag, or 40-char commit SHA). `MODEL_CARD.md` documents the recommended pre-v1 procedure: download upstream at a known revision, capture the commit SHA, pin it, re-run baselines. Strict-local mode does not cover HF downloads — that's now explicit. (Audit §4 MEDIUM.)
- **README rewritten problem-first.** New "Why this exists," "Why fine-tune instead of base Qwen + RAG?," and "What 'done' looks like" sections answer the first three questions a reviewer is going to ask. (Audit §6, §7 MEDIUM.)
- **Governance hardening.** `GOVERNANCE.md` adds an issue-triage label table (`bug`, `enhancement`, `recipe`, `eval`, `train`, `docs`, `security`, `good first issue`, `help wanted`, `wontfix`, `needs-repro`), a communication-channels section (deliberately small surface area pre-1.0), and a trademark / naming-clearance posture (project-name search recorded as 2026-04-25; "civic-slm" forks must be visibly marked, can't publish under confusable names). City names that appear in the corpus are now explicitly disclaimed as non-affiliations. (Audit §5, §8, §10 MEDIUM.)
- **CI vs. local clarified.** `CONTRIBUTING.md` now states explicitly that CI on Linux skips the `train` extra (MLX is Apple-Silicon-only), so a green Linux CI on a training-touching diff is a known false-positive — local dry-run is required. Supply-chain posture (7-day min-release-age, `ignore-scripts=true`) documented for forkers. (Audit §2, §4 MEDIUM.)
- **California-isms documented.** `docs/RECIPES.md` opens with a banner spelling out that Texas / NY / Ohio jurisdictions use different vocabulary (Specific Use Permits vs. CUPs; SEQRA vs. CEQA; comprehensive plan vs. general plan) and that recipe authors must adjust the prompts to local vocabulary. The schema is neutral; the prompts are not. (Audit §11 MEDIUM.)
- **`docs/USAGE.md` no longer hardcodes the maintainer's `~/Projects/...` path** — the walkthrough now starts with `git clone` so users following it verbatim succeed. (Audit §2 LOW.)

### Added — MEDIUM tier from `AUDIT.md`

- **Pydantic-validated training configs.** `TrainConfig` is now a frozen Pydantic model with stage-aware invariants (CPT must set `train.iters`; SFT/DPO must set `train.epochs`; DPO must set `train.beta`). A typo in `configs/sft.yaml` previously surfaced as a `KeyError` deep inside `build_command`; it now raises `ConfigError` at load time with a message that points back at the canonical example configs.
- **Hyperparameter rationale in every config.** `configs/{cpt,sft,dpo}.yaml` now carry inline comments explaining each value: why r=64/α=128 for CPT vs. r=32/α=64 for SFT/DPO, why LR ladders by two orders of magnitude across stages, why DPO uses zero dropout, what each warmup choice is buying. Reviewers no longer have to chase down "why this number?" out-of-band.
- **Synth idempotency.** `generate_corpus()` reads `out_path` on entry and skips chunk+task pairs that already produced examples. Re-running an interrupted ~$15 synth job is now ~$0; a fresh run still works the same. Opt out with `resume=False`.
- **`RELEASING.md`** documents the full release checklist, the SemVer policy (with the project-specific clarification on eval-harness changes), and a draft post-1.0 deprecation policy.
- **DCO sign-off** is now required for contributions. `CONTRIBUTING.md` documents `git commit -s` and `git rebase --signoff main` for older branches.
- **`docs/GLOSSARY.md`** — plain-language definitions of ML terms (LoRA, CPT, SFT, DPO, quantization) and civic terms (Brown Act, CEQA, CUP, comprehensive plan, public-records statute) for the non-ML civic technologists the project targets. Linked from README.
- **Data-flow diagram in `ARCHITECTURE.md`** showing how SHA-256 propagates `CivicDocument → DocumentChunk → Provenance → EvalExample` and how the contamination check binds to it. Plus three stage-boundary invariants stated explicitly.
- **`examples/` directory** with three copy-paste-runnable demos: ask-a-question, run-the-factuality-eval, inspect-a-baseline. The third one runs without a server — it's the lowest-friction "what does this thing do?" path.

### Added — open-source-readiness pass (BLOCKERs + HIGHs from `AUDIT.md`)

- **License reconciled.** `pyproject.toml` now declares `license = { text = "MIT" }` to match `LICENSE`, and `version` is sourced from the `VERSION` file via `tool.hatch.version` so the published package can no longer disagree with the repo. `civic_slm.__version__` resolves through `importlib.metadata` with a `VERSION`-file fallback for editable checkouts.
- **Description and package docstring updated** to reflect the all-50-states scope (was "California municipal documents").
- **`MODEL_CARD.md`, `DATA_CARD.md`, `ACCEPTABLE_USE_POLICY.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`, `ROADMAP.md`, `CITATION.cff`** all added at the repo root. Model card calls out California-shaped training, regex/word-overlap scorer limitations, no multi-seed runs, and the v0.0.1 refusal-benchmark caveat. Data card spells out the per-source license-audit gate.
- **`docs/SOURCES.md`** is the new gate for ingesting any real civic document. The `san-clemente` entry is filled in as `NO-GO until terms-of-use clause is captured` so v0.1.0 cannot accidentally crawl real bytes.
- **Issue and PR templates** under `.github/`: bug report, recipe request, PR checklist (including a privacy/safety self-check).
- **Refusal benchmark fixed.** `data/eval/refusal.jsonl` previously contained only `expected_refusal: true` examples — a model that always refuses scored 1.0. Added `r011`–`r014`, four should-answer negatives, so the scorer can now distinguish refusal recall from over-refusal precision. Pre-v0.1.0 base-Qwen refusal numbers are not comparable.
- **Train/eval contamination check.** `Provenance.source_doc_hash` and `_EvalBase.source_doc_hash` (both optional, SHA-256) added to the schema; `DocumentChunk` now carries `source_doc_hash` so synth can populate `Provenance.source_doc_hash` automatically. `civic_slm.eval.runner.assert_no_contamination()` raises `ContaminationError` at startup if any eval example's source-doc hash appears in `data/raw/manifest.jsonl`. Override is opt-in via `--allow-contamination` and logs loudly. Synthetic-only evals with `source_doc_hash: null` pass trivially.
- **Transcript PII scrubbing.** `src/civic_slm/ingest/video/caption.py` now scrubs speaker labels to `[Speaker]` by default and redacts U.S. street-address-shaped substrings to `[ADDRESS]`. Public-comment blocks (anything between a `>> Public Comment` header and the next non-public-comment section header) **always** strip speaker labels regardless of opt-out. Set `CIVIC_SLM_KEEP_SPEAKER_NAMES=1` to retain non-public-comment speakers (intended for elected-officials-only contexts).
- **Eval reproducibility.** `civic-slm eval run` now records `seed`, `temperature`, `max_tokens`, served-model id, base URL, civic-slm version, and example count in a `_run_config` header on the JSONL output and at the top of the markdown report. `ChatClient` accepts a `seed=` kwarg and sends it to the OpenAI-compatible endpoint.
- **README banner** declares the v0.1.0 release as an _infrastructure preview_ — no fine-tuned model has shipped yet, the only registered recipe is `san-clemente`, and `docs/SOURCES.md` is the gate before any real crawl.

### Added

- **Strict-local mode — zero API spend, with proof.** Set `CIVIC_SLM_STRICT_LOCAL=1` to make synth, the side-by-side judge, and the browser-use crawler **refuse to use Anthropic** at runtime. Misconfigured env? They raise loudly instead of silently spending tokens. Pair it with `civic-slm doctor --strict-local` for a one-shot audit that exits non-zero if any code path could reach a paid endpoint — checks the backend env, the loaded secrets, and pings both candidate and teacher URLs. Runs in seconds. Useful before a multi-hour synth job or on any fresh machine where you don't want surprises.

### For contributors

- New `civic_slm.serve.runtimes.is_strict_local()` helper — single source of truth for `CIVIC_SLM_STRICT_LOCAL` parsing (truthy: `1|true|yes|on`).
- `civic_slm.llm.backend.select_backend()` and `civic_slm.ingest.recipes._browser.agent_llm()` (renamed from `_agent_llm`) consult `is_strict_local()` and raise `RuntimeError` if the resolved backend would be Anthropic.
- `civic-slm doctor` adds `--strict-local`: forces backend=`local`, fails on a loaded `ANTHROPIC_API_KEY`, hardens unreachable-teacher to fail (was warn), and soft-warns on candidate/teacher URLs that don't look local (`127.0.0.1`, `localhost`, `*.local`, RFC 1918).
- New `tests/test_strict_local.py` — 21 cases across the env helper, the two runtime tripwires, and the doctor exit codes.
- `docs/RUNTIMES.md`: new "Strict-local mode" section with the verified-zero-spend recipe; env table now lists `CIVIC_SLM_STRICT_LOCAL`, `CIVIC_SLM_TIMEOUT_S`, and `CIVIC_SLM_WHISPER_MODEL`.

### Added

- **Meeting video / transcript ingestion.** `civic-slm crawl-videos <slug>` discovers council-meeting videos from YouTube channels or playlists, fetches audio + captions, extracts a transcript with a caption-first priority chain (human SRT/VTT → YouTube auto-caption → Whisper ASR fallback), and lands a `MEETING_TRANSCRIPT` row alongside everything else in `data/raw/manifest.jsonl`. Speaker labels are preserved heuristically from VTT `<v Name>` voice tags and `>> NAME:` close-caption patterns; full diarization is a v1 line item.

### For contributors

- New `DocType.MEETING_TRANSCRIPT` plus three optional fields on `CivicDocument`: `video_url`, `transcript_source` (`human_srt | youtube_caption | vtt | whisper`), and `duration_s`. Fully optional — existing manifest entries unaffected.
- New module `src/civic_slm/ingest/video/` with `caption.py` (VTT/SRT parser + rolling cue dedup that handles YouTube's growing-cue auto-caption pattern), `youtube.py` (yt-dlp wrapper for channel enumeration + audio/caption fetch), `transcript.py` (caption-first orchestrator), and `asr.py` (lazy mlx-whisper wrapper).
- New `Recipe.discover_videos()` _optional_ method + `crawl_videos()` orchestrator next to `crawl()`. Recipes that don't implement it are skipped silently.
- New shared helpers `recipes/_youtube.py`: `youtube_channel_videos(channel_url, since, max_videos)` and `youtube_playlist_videos(playlist_url, max_videos)` — plumbing for any new recipe.
- 12 new tests across `tests/test_caption.py` (8 cases covering VTT/SRT parsing, voice-tag preservation, rolling-cue dedup, paragraph breaks on speaker change) and `tests/test_video_ingest.py` (4 cases covering crawl_videos with stubbed yt-dlp + ASR — idempotent re-runs, no-video-support recipes, empty-transcript skip).
- `pyproject.toml` `ingest` extra adds `yt-dlp>=2025.1` and `mlx-whisper>=0.4` (the latter gated to Apple Silicon).
- `CIVIC_SLM_WHISPER_MODEL` env knob — defaults to `mlx-community/whisper-large-v3-turbo`.

## [0.1.0] - 2026-04-24

The pipeline is now portable across all 50 U.S. states and runtime-agnostic. San Clemente, CA stays as the demo; everything else generalizes.

### Added

- **Runtime-agnostic serving.** Eval, side-by-side, and synth all read `CIVIC_SLM_CANDIDATE_URL` / `CIVIC_SLM_CANDIDATE_MODEL` (and `_TEACHER_*` for the comparator/teacher) so you can serve via MLX-LM, Ollama, LM Studio, llama.cpp, vLLM, or any OpenAI-compatible endpoint without code changes. New `docs/RUNTIMES.md` has copy-paste setup per runtime, plus a streamlined model matrix (1 model minimum + Anthropic, or 2 models for fully-local).
- **`civic-slm doctor` command.** Pings configured candidate + teacher URLs, validates secrets, and prints a single status table. Run it before any other stage when you're not sure what's wired up.
- **Generalized to all 50 states.** Project is no longer California-specific. Crawl any U.S. city, county, or township with a one-file recipe — see [`docs/RECIPES.md`](docs/RECIPES.md) and the new `recipes/_template.py`. San Clemente, CA stays as the demo recipe.
- **Fully-local LLM backend.** Synth, the side-by-side judge, and the browser-use crawler now route through a single backend abstraction (`civic_slm.llm.backend`). Set `CIVIC_SLM_LLM_BACKEND=local` (with `CIVIC_SLM_LOCAL_LLM_URL` and `CIVIC_SLM_LOCAL_LLM_MODEL`) to run the whole pipeline against a locally served OpenAI-compatible endpoint — no Anthropic, no external APIs required. Default behavior is unchanged.

### Changed (breaking)

- `CivicDocument.city` → `jurisdiction` (covers cities, counties, townships, school districts).
- `CivicDocument` requires a new `state` field — 2-letter U.S. postal code (`CA`, `TX`, `NY`, ...).
- Recipe Protocol: `Recipe.city` → `Recipe.jurisdiction` + `Recipe.state`.
- CLI: `civic-slm crawl --city <slug>` → `--jurisdiction <slug>` (with `--city` retained as an alias for the demo).
- `synth.generate.generate_for_chunk` and `generate_corpus` now take `jurisdiction` + `state` instead of `city`. Synth prompt templates (`prompts/*.md`) substitute `{jurisdiction}` and `{state}`.
- `data/raw/` layout: now `data/raw/<state-lower>/<jurisdiction>/<meeting-date>/<file>` (was `data/raw/<city>/<meeting-date>/<file>`).
- `DocType` expanded: added `comprehensive_plan`, `master_plan`, `zoning_ordinance`, `ordinance`, `resolution`, `budget`, `rfp`, `notice`. Existing values (`general_plan`, `staff_report`, `minutes`, `agenda`, `municipal_code`, `other`) unchanged.
- `side_by_side.jsonl`: 10 prompts replaced with U.S.-broad questions (NEPA, TIF, comprehensive plans, open-meetings law) instead of CA-only ones (CEQA-only, Mello-Roos, Brown Act). Examples acknowledge state-by-state variation.
- Judge system prompt: "California municipal government" → "U.S. local government" with state-specific accuracy weighted heavily when the prompt names one.

### For contributors

- New module `src/civic_slm/llm/backend.py` with `Backend` Protocol, `LocalBackend` (httpx → /v1/chat/completions), `AnthropicBackend` (lazy SDK import), and `select_backend()` env dispatch.
- New module `src/civic_slm/serve/runtimes.py` with `Runtime` enum + per-runtime presets (MLX, llama.cpp, Ollama, LM Studio, OpenAI-compatible) and env-driven `candidate_url() / candidate_model() / teacher_url() / teacher_model()` helpers.
- New `src/civic_slm/doctor.py` (wired into umbrella CLI as `civic-slm doctor`): pings configured candidate + teacher URLs, validates secrets, prints rich status table.
- `synth.generate.generate_for_chunk` and `generate_corpus` accept an optional `backend=` param; default resolves from env. They now take `jurisdiction` + `state` instead of `city`.
- `eval.runner` and `eval.side_by_side` default `--base-url` / `--served-model` to `CIVIC_SLM_CANDIDATE_URL` / `CIVIC_SLM_CANDIDATE_MODEL` (and `_TEACHER_*` for the comparator).
- `eval.judge.judge_pair` and `judge_with_position_swap` accept an optional `backend=` param.
- `SanClementeRecipe.discover` (and the new `recipes/_template.py`) select `ChatAnthropic` or `ChatOpenAI` based on `CIVIC_SLM_LLM_BACKEND`.
- New: `src/civic_slm/ingest/recipes/_template.py` (annotated skeleton) and `docs/RECIPES.md` (step-by-step add-a-jurisdiction walkthrough). Plus `docs/RUNTIMES.md` (copy-paste setup per runtime, streamlined model matrix).
- New tests: `test_backend.py` (4 cases: env dispatch, unknown-backend rejection, mocked-transport OpenAI payload assertion) and `test_civic_document_rejects_bad_state` (rejects "California" — must be 2-letter postal code).
- README, ARCHITECTURE, CONTRIBUTING, USAGE, and CLAUDE.md all updated to U.S.-wide framing and runtime-agnostic guidance.

## [0.0.1] - 2026-04-24

First baseline. The fine-tune pipeline isn't trained yet, but every step it depends on is wired and tested, and the bars training has to clear are committed to the repo.

### What you can do now

- Run a four-bench evaluation against any OpenAI-compatible chat endpoint with `civic-slm eval run`. Reports land in `artifacts/evals/<model>/<bench>.{json,md}`.
- Crawl city websites with `civic-slm crawl --city san-clemente`, driven by an LLM-controlled browser instead of platform-specific scrapers. Idempotent across re-runs (manifest deduped by sha256).
- Generate synthetic SFT pairs from real document chunks via the Anthropic SDK and a per-task prompt taxonomy (`qa_grounded`, `refusal`, `extract`, `summarize`); every example schema-validated before landing in `data/sft/`.
- Train CPT, SFT, and DPO stages via `civic-slm train {cpt,sft,sft}` (delegates to `mlx_lm.lora` / `mlx_lm.dpo`).
- Fuse a final adapter and export both MLX 4-bit and GGUF Q5_K_M with `civic-slm merge`.
- Review synthetic SFT examples in a terminal accept/reject loop with `civic-slm review-sft`.

### Baseline numbers (Qwen2.5-7B-Instruct, MLX 4-bit)

| Bench      | n   | Mean  | Median | Latency |
| ---------- | --- | ----- | ------ | ------- |
| factuality | 10  | 0.501 | 0.566  | 637 ms  |
| refusal    | 10  | 0.800 | 1.000  | 460 ms  |
| extraction | 5   | 0.277 | 0.000  | 925 ms  |

`side_by_side` is wired (Claude Sonnet 4.6 judge with A/B position swap) but pending a 72B GGUF comparator on llama.cpp.

### For contributors

- Apple Silicon stack: MLX-LM for training and serving, llama.cpp for GGUF + the 72B comparator. No CUDA assumed.
- 37 tests across schema, ingest end-to-end (with stub fetcher + idempotency), PDF chunker, scorers, eval runner, judge parser, synth parser, and train command builders.
- Strict pyright, ruff, ruff-format. `pyproject.toml` extras: `ingest` (browser-use, pypdf), `synth` (anthropic), `train` (mlx, mlx-lm, datasets, wandb), `eval` (sentence-transformers).
- Eval contract is **eval-first**: do not start training until any change to the harness still reproduces the baseline numbers above.

[Unreleased]: https://github.com/itsmeduncan/civic-slm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/itsmeduncan/civic-slm/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/itsmeduncan/civic-slm/releases/tag/v0.0.1
