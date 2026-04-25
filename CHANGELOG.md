# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — v0.2.x Track A1: BGE-reranker factuality scorer

- **`civic-slm eval run --similarity {word_overlap,bge}`.** New flag. Default stays `word_overlap` so pre-v0.2 baselines remain bit-reproducible; opt into `bge` to get BAAI/bge-large-en-v1.5 dual-encoder cosine, mapped to `[0, 1]`. The choice is recorded in the eval JSONL `_run_config` header alongside `bge_model`.
- **`civic_slm.eval.embeddings.bge_similarity_fn(model_id=...)`** — lazy-loaded helper, caches the encoder in module state. The encoder is only imported when `--similarity bge` is selected, so the default install does not pull `sentence-transformers` into the import graph.
- **`civic-slm eval run` records `similarity` and `bge_model`** in the run config header, so a markdown report immediately tells you which scorer produced the numbers.
- **Tests:** `tests/test_embeddings.py` covers the `_resolve_similarity` mapping and an empty-input short-circuit; the actual model-download check is gated behind `CIVIC_SLM_RUN_BGE_TEST=1` so the default `pytest` stays fast and offline.

### Breaking — eval

- **Factuality numbers under `--similarity bge` are not comparable to numbers under `--similarity word_overlap`.** They use different scales. Pre-v0.2 baselines in `artifacts/evals/base-qwen2.5-7b/factuality.{json,md}` were produced under word-overlap and remain valid for that scorer. v0.2 baselines under BGE will be added in a separate file (`factuality.bge.{json,md}`) when the maintainer re-runs them; until then, **do not mix the two**. Reports now record `similarity:` so this is auditable per-run.

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

- **Meeting video / transcript ingestion.** `civic-slm crawl-videos --jurisdiction <slug>` discovers council-meeting videos from YouTube channels or playlists, fetches audio + captions, extracts a transcript with a caption-first priority chain (human SRT/VTT → YouTube auto-caption → Whisper ASR fallback), and lands a `MEETING_TRANSCRIPT` row alongside everything else in `data/raw/manifest.jsonl`. Speaker labels are preserved heuristically from VTT `<v Name>` voice tags and `>> NAME:` close-caption patterns; full diarization is a v1 line item.

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
- Fuse a final adapter and export both MLX 4-bit and GGUF Q5_K_M with `python scripts/merge_quantize.py`.
- Review synthetic SFT examples in a terminal accept/reject loop with `python scripts/review_sft.py`.

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
