# Roadmap

> The roadmap is the maintainer's current best guess. Things move. Open an
> issue if you want to influence the order.

## v0.1.x — infrastructure preview (current)

Status: shipped on 2026-04-25. No fine-tuned model yet; what's in the repo
is the pipeline that will produce one.

- [x] Crawl + chunk + manifest.
- [x] Synth pipeline against Claude (and local fallback).
- [x] Eval harness (factuality, refusal, extraction, side-by-side).
- [x] CPT / SFT / DPO training wrappers (MLX-LM).
- [x] Merge + quantize to MLX 4-bit and GGUF Q5_K_M.
- [x] Strict-local mode with runtime tripwires.
- [x] Open-source-readiness pass: license, model card, data card, AUP, CoC,
      governance, transcript PII scrubbing, train/eval contamination check,
      refusal-eval should-answer negatives.

## v0.2.x — first real corpus and first fine-tune

Goal: a tagged civic-slm-v1 model on HF Hub that beats base Qwen2.5-7B on
≥3/4 benchmarks at the v1 sample sizes.

- [ ] Fill `docs/SOURCES.md` for `san-clemente`. First real crawl.
- [ ] Generate ~5,000 SFT examples; human-review the first 500 with `scripts/review_sft.py`.
- [ ] Eval scale-up: 200 / 100 / 50 / 100 (factuality / refusal / extraction / side-by-side).
- [ ] BGE-reranker swap of the factuality scorer.
- [ ] CPT smoke run, then real CPT, then SFT, then DPO.
- [ ] Merge + quantize to v1 artifacts.
- [ ] Publish v1 model card and data card alongside weights.

## v0.3.x — generalization across U.S. jurisdictions

Goal: the "all 50 states" framing in the README becomes load-bearing.

- [ ] Add 5–10 recipes across regions and platforms (Granicus, Legistar,
      CivicPlus, Municode, vendor-free).
- [ ] Second-city held-out eval (e.g., Austin TX, Cuyahoga County OH).
- [ ] Per-jurisdiction distribution reported in the data card.
- [ ] 72B GGUF comparator wired into `side_by_side`.

## v1.0 — long-term-supported release

Goal: backward-compatibility commitment and a stable release cadence.

- [ ] Deprecation policy.
- [ ] Sigstore / artifact signing.
- [ ] SBOM (CycloneDX or SPDX) emitted from CI.
- [ ] Citation-format file (CITATION.cff).
