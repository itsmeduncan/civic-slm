<!-- Thanks for contributing. Please fill out each section. -->

## What this changes

<!-- One paragraph: what shipped, why. Don't repeat the diff. -->

## Affected stages

<!-- Check all that apply. -->

- [ ] crawl (PDF)
- [ ] crawl (video / transcripts)
- [ ] synth
- [ ] eval
- [ ] train (CPT / SFT / DPO)
- [ ] merge / quantize
- [ ] serve
- [ ] docs / DX
- [ ] CI / release

## Required for merge

- [ ] `uv run ruff check .` clean
- [ ] `uv run ruff format --check .` clean
- [ ] `uv run pyright` clean (strict mode)
- [ ] `uv run pytest` green
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If this adds a new recipe: `docs/SOURCES.md` audit entry filled in
- [ ] If this changes the eval harness: baselines in `artifacts/evals/base-qwen2.5-7b/` re-run and committed (or the regression is explained)
- [ ] If this changes a Pydantic schema: migration noted, frozen-model tests still pass

## Privacy / safety self-check

- [ ] No real public-commenter PII (names, addresses, phone numbers) in any committed file
- [ ] No real API keys, tokens, or secrets in the diff
- [ ] If this touches the synth or crawl pipeline: behavior under `CIVIC_SLM_STRICT_LOCAL=1` is preserved (or the change is explicitly out-of-scope and noted)

## Notes for the reviewer

<!-- Anything the reviewer should look at first. Performance numbers, screenshots,
     surprising decisions, follow-ups deferred to other PRs, etc. -->
