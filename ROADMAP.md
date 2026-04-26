# Roadmap

> The roadmap is the maintainer's current best guess. Things move. Open an
> issue if you want to influence the order.

**Legend:** `[x]` shipped, `[ ]` not started, `[ ] **Partial.**` partially
shipped (the parenthetical describes what landed and what's still open).

## v0.1.x — infrastructure preview (shipped 2026-04-25)

The pipeline that will produce a fine-tune. No fine-tuned model in this
release.

- [x] Crawl + chunk + manifest.
- [x] Synth pipeline against Claude (and local fallback).
- [x] Eval harness (factuality, refusal, extraction, side-by-side).
- [x] CPT / SFT / DPO training wrappers (MLX-LM).
- [x] Merge + quantize to MLX 4-bit and GGUF Q5_K_M.
- [x] Strict-local mode with runtime tripwires.
- [x] Open-source-readiness pass: license, model card, data card, AUP, CoC,
      governance, transcript PII scrubbing, train/eval contamination check,
      refusal-eval should-answer negatives.

## v0.2.x — first real corpus and first fine-tune (in progress)

Goal: a tagged civic-slm-v1 model on HF Hub that beats base Qwen2.5-7B on
≥ 3/4 benchmarks at v1 sample sizes (200 / 100 / 50 / 100).

**Current status (as of 2026-04-25).** All four code-only prereq tracks have
landed on `main`: PR #6 (BGE scorer), PR #7 (training-pipeline robustness),
PR #8 (72B comparator wiring), PR #9 (eval scale-up + multi-jurisdiction
seeding), PR #10 (post-ship doc sync). What remains is **maintainer-blocked**
— the ToS audit, real crawl, synth corpus, actual training runs, and HF Hub
publish.

Code prereqs (landed in v0.2.x):

- [x] BGE-reranker swap of the factuality scorer
      (`civic-slm eval run --similarity bge`; opt-in, default unchanged).
- [x] Training-pipeline robustness: `--smoke-test`, `--resume` guard, and a
      signal-aware subprocess supervisor that flushes a checkpoint on Ctrl-C.
- [x] 72B GGUF comparator wiring for `side_by_side`: `civic-slm doctor
    --teacher` ping, `ComparatorMissingError` fail-fast, runbook in
      `docs/RUNTIMES.md`. Maintainer must download a 72B GGUF and run
      `llama-server` to actually produce numbers.
- [ ] **Partial.** Eval scale-up: target 200 / 100 / 50 / 100, currently
      at 25 / 29 / 15 / 25 with multi-jurisdiction seeding (Austin TX,
      Houston TX, NYC, Phoenix AZ, Seattle WA, Cook County IL, Cuyahoga
      County OH, Atlanta GA, Boston MA, Denver CO, Portland OR). Further
      authoring + bench grown from real corpus chunks lands incrementally.

Maintainer-blocked (gates the rest of v0.2):

- [ ] Re-baseline base Qwen on the v0.2 bench. Needs `mlx_lm.server` running.
      Without this, the `_re-baselining_` placeholders in `MODEL_CARD.md`
      stay; with it, every later stage has a real target to beat.
- [ ] Fill `docs/SOURCES.md` for `san-clemente` — capture the verbatim
      terms-of-use clause and flip the audit to GO. Blocks the first real
      crawl.
- [ ] First real crawl: `civic-slm crawl --jurisdiction san-clemente
    --max 50` plus `crawl-videos --max 20`.
- [ ] Generate ~5,000 SFT examples (`civic_slm.synth.generate.generate_corpus`,
      ~$5–15 in Anthropic credits); human-review the first 500 with
      `python scripts/review_sft.py`.
- [ ] CPT smoke run → full CPT → SFT → DPO. Code is ready (see Track A2
      above); needs the corpus first.
- [ ] Merge + quantize to v1 artifacts (`scripts/merge_quantize.py`).
- [ ] Publish v1 model card and data card alongside weights, push to HF Hub,
      tag `v0.2.0` per `RELEASING.md`.

## v0.3.x — generalization across U.S. jurisdictions

Goal: the "all 50 states" framing in the README becomes load-bearing.

- [ ] Add 5–10 recipes across regions and platforms (Granicus, Legistar,
      CivicPlus, Municode, vendor-free).
- [ ] Second-city held-out eval against trained model weights (e.g.,
      Austin TX, Cuyahoga County OH). v0.2 seeded examples from those
      jurisdictions in the eval harness; this validates against actual
      trained model output.
- [ ] Per-jurisdiction distribution reported in the data card.

## v1.0 — long-term-supported release

Goal: backward-compatibility commitment and a stable release cadence.

- [x] Deprecation policy (drafted in `RELEASING.md`; binds at v1.0).
- [x] Citation-format file (`CITATION.cff` shipped in v0.1.0).
- [ ] Sigstore / artifact signing.
- [ ] SBOM (CycloneDX or SPDX) emitted from CI.
