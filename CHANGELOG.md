# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- New: `src/civic_slm/ingest/recipes/_template.py` (annotated skeleton) and `docs/RECIPES.md` (step-by-step add-a-jurisdiction walkthrough).
- New tests: `test_civic_document_rejects_bad_state` (rejects "California" — must be 2-letter postal code).
- README, ARCHITECTURE, CONTRIBUTING, USAGE, and CLAUDE.md all updated to U.S.-wide framing.

### For contributors

- New module `src/civic_slm/llm/backend.py` with `Backend` Protocol, `LocalBackend` (httpx → /v1/chat/completions), `AnthropicBackend` (lazy SDK import), and `select_backend()` env dispatch.
- `synth.generate.generate_for_chunk` and `generate_corpus` accept an optional `backend=` param; default resolves from env.
- `eval.judge.judge_pair` and `judge_with_position_swap` accept an optional `backend=` param.
- `SanClementeRecipe.discover` selects `ChatAnthropic` or `ChatOpenAI` based on `CIVIC_SLM_LLM_BACKEND`.
- 4 new tests (`tests/test_backend.py`) covering env dispatch, unknown backend rejection, and a mocked-transport assertion that `LocalBackend` posts the OpenAI-compatible payload.

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

[Unreleased]: https://github.com/itsmeduncan/civic-slm/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/itsmeduncan/civic-slm/releases/tag/v0.0.1
