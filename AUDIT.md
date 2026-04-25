# civic-slm — Pre-Open-Source Release Audit

**Audited:** 2026-04-25 against repo HEAD (VERSION 0.1.0).
**Scope:** Read-only, evidence-grounded, 12-stakeholder review.
**Mandate:** findings only; no fixes, no reformatting.

Severity legend: `BLOCKER` (ship-stopper) · `HIGH` (within 2 weeks of launch) · `MEDIUM` (next quarter) · `LOW` (nice-to-have) · `INFO` (note for awareness).

---

## 1. Principal Engineer

- **[BLOCKER]** **License declaration is internally inconsistent.** `pyproject.toml:8` declares `license = { text = "Apache-2.0" }`, but `LICENSE:1` is `MIT License`. PyPI/distros and HF metadata will pick up Apache-2.0; the file in the repo is MIT. _Why it matters:_ this is a legal-release blocker — the package literally ships with two different license claims, and downstream redistribution will be ambiguous.
- **[HIGH]** **Package version is hardcoded to a stale value.** `pyproject.toml:3` is `version = "0.0.0"`, while `VERSION` is `0.1.0` and `CHANGELOG.md` already documents a v0.1.0 release. _Why it matters:_ users `pip install civic-slm` will get a `0.0.0` build; `python -c "import civic_slm; print(civic_slm.__version__)"` will lie about the release. There is no `dynamic = ["version"]` wiring to the `VERSION` file.
- **[HIGH]** **Package description is stale and contradicts shipped scope.** `pyproject.toml:4` says "California municipal documents." `README.md:3` and `CHANGELOG.md` (v0.1.0) explicitly claim all-50-states U.S. local government. _Why it matters:_ PyPI listing and search indexing will be wrong on day one.
- **[HIGH]** **Public package docstring is stale.** `src/civic_slm/__init__.py:1` still describes the project as "for CA municipal documents," contradicting `README.md:3`. _Why it matters:_ first thing a user sees in `help(civic_slm)` is wrong.
- **[MEDIUM]** **Training-config access is unguarded against missing keys.** `src/civic_slm/train/sft.py:24-29` and `train/cpt.py:33-…` index nested YAML directly (`raw["data"]["train_path"]`, `raw["lora"]["rank"]`) with no `.get()` defaults or upfront schema validation. _Why it matters:_ a single typo in `configs/sft.yaml` becomes an unhelpful `KeyError`, not the actionable error the rest of the codebase enforces (cf. `config.py:74`).
- **[MEDIUM]** **`TrainConfig` skips its own validation pattern.** `src/civic_slm/train/common.py:20-42` loads YAML into a raw dict instead of a Pydantic model — inconsistent with `ARCHITECTURE.md:82-83` ("All models are `frozen=True, extra="forbid"`"). _Why it matters:_ contract drift between the training stage and the rest of the data pipeline.
- **[LOW]** **No checkpoint resume in CPT/SFT/DPO wrappers.** `src/civic_slm/train/{cpt,sft,dpo}.py` shell out to `mlx_lm` with no `--resume-from`/`--checkpoint-path` flag exposed. _Why it matters:_ a 2000-iter CPT run that OOMs on iter 1900 starts from scratch.
- **[LOW]** **`eval/side_by_side.py` has no direct unit test.** It depends on `judge.py` (which is tested) but its own runner logic is uncovered. _Why it matters:_ side-by-side is the headline benchmark vs. base/72B — silent breakage would be embarrassing.
- **[INFO]** Type-checking is strict and clean; no TODO/FIXME/HACK/XXX in `src/` or `scripts/`; no swallowed exceptions outside intentional, commented sites (e.g., `train/common.py:71-72,80`, `eval/judge.py:120-121`). Public APIs are typed throughout.

## 2. SRE / Infrastructure

- **[HIGH]** **No graceful failure on partial training runs.** No checkpoint resume (above); no signal-handling for SIGTERM/SIGINT in `train/cpt.py`/`train/sft.py`. _Why it matters:_ a long CPT run interrupted by laptop sleep loses everything.
- **[HIGH]** **W&B observability is best-effort and silent on failure.** `src/civic_slm/train/common.py:71-80` swallows wandb init exceptions and continues. _Why it matters:_ a multi-hour run can complete with no metrics history because of an unrelated transient — and you won't know until the run finishes.
- **[MEDIUM]** **CI matrix runs Linux but training cannot run on Linux.** `.github/workflows/ci.yml:19-44` excludes the `train` extra on Linux. That's correct, but no platform-coverage badge or doc says "tests pass on Linux ≠ training works on Linux." _Why it matters:_ contributors with Linux CI greens may assume their training-touching change is verified.
- **[MEDIUM]** **No idempotent re-run guarantee for `synth.generate.generate_corpus`.** `src/civic_slm/synth/generate.py` records provenance (lines 94-99) but does not skip a chunk that already produced an example. _Why it matters:_ partial-progress synth runs (a `~$5–15` job per `docs/USAGE.md:155-182`) can re-bill on retry.
- **[MEDIUM]** **OOM handling is implicit.** No automatic batch-size reduction, no `--gradient-checkpointing` exposed in CPT (only SFT, `configs/sft.yaml`). _Why it matters:_ unified-memory pressure on a smaller Mac (≤32GB) means the user discovers OOM on iter ~50, not at config time.
- **[LOW]** **Hardcoded dev path in walkthrough.** `docs/USAGE.md:8` says `cd ~/Projects/src/github.com/itsmeduncan/civic-slm`. _Why it matters:_ harmless but signals "this was written for the author."
- **[INFO]** No hardcoded `/Users/...` in `src/`. Secrets are read from `~/.config/civic-slm/.env` and never logged (`config.py:39-76`, `doctor.py:130-141`). Crawl is content-addressed by sha256 (`ingest/manifest.py:36-37`).

## 3. ML Research Engineer

- **[BLOCKER]** **The refusal benchmark is structurally degenerate.** All 10 examples in `data/eval/refusal.jsonl` have `expected_refusal: true`. _Evidence:_ `data/eval/refusal.jsonl:1-10` (verified: `total: 10, true: 10, false: 0`). The scorer in `src/civic_slm/eval/scorers.py:94` is `correct = refused == example.expected_refusal`. _Why it matters:_ a model that **always refuses every input** scores 1.0 on this benchmark. The reported base score of 0.800 (`artifacts/evals/base-qwen2.5-7b/refusal.json`) just means the base model failed to refuse twice. Any "civic intelligence vs. civic hallucination" claim built on this number is unsupported until the eval contains a ≥30% should-answer negative class. **This is the safety story for the entire project; until fixed, do not publish refusal numbers.**
- **[BLOCKER]** **No contamination check between training and eval source documents.** `Provenance` (`src/civic_slm/schema.py:113-120`) records `generator`, `model`, `prompt_sha`, `created_at` — but **no `source_doc_hash`**. `synth/generate.py:94-99` populates exactly those fields. `eval/runner.py:49-150` loads `EvalExample` JSONL with no schema field for source-document provenance and no runtime check against any train manifest. _Why it matters:_ once `data/raw/` and `data/sft/` are populated (currently empty), there is **no mechanism, code or schema, that prevents an eval document from also being in the training set.** CLAUDE.md (the audit prompt) explicitly designates this a BLOCKER.
- **[HIGH]** **Eval sample sizes are below the threshold for any defensible claim.** `wc -l data/eval/*.jsonl`: factuality 10, refusal 10, side*by_side 10, structured_extraction 5. \_Why it matters:* even with a clean methodology, n=5/10 means a one-example flip moves the mean by 10–20 points. CLAUDE.md targets v1 at 200/100/50/100; until then, no headline claim should be published.
- **[HIGH]** **No documented seeds, no multiple-seed runs, no confidence intervals.** `eval/runner.py` records `latency_ms` but nothing in `artifacts/evals/base-qwen2.5-7b/*.json` includes a seed, a temperature, a sampling config, or a CI. _Why it matters:_ "0.501" reported without seed/temp is irreproducible.
- **[HIGH]** **No second-city / out-of-distribution generalization evaluation.** All eval contexts are San Clemente-flavored synthetic snippets (`data/eval/civic_factuality.jsonl`, `structured_extraction.jsonl` — councilmembers Ramirez/Chen/Patel/Whitfield, El Camino Real). _Why it matters:_ the project markets a 50-state model (`README.md:3`); without an Austin-TX/Cuyahoga-County held-out, the generalization claim is aspirational.
- **[HIGH]** **Refusal scorer is regex-only and easy to game.** `src/civic_slm/eval/scorers.py:31-38, 86-103` matches a small set of phrases (`I don't know`, `cannot find/determine/answer`, …). _Why it matters:_ a model that says "The provided context does not contain that information; I will not speculate" might still match (it does, via "not …in"), but a phrase like "I am unsure" does **not** match. The scoring is brittle and underspecified relative to the safety claim.
- **[HIGH]** **Factuality scorer is word-overlap, not semantic.** `ARCHITECTURE.md:62-67` admits "BGE reranker swap planned" — but the swap hasn't happened, and `CLAUDE.md` lists it as the next step. _Why it matters:_ word-overlap rewards verbatim copying and penalizes correct paraphrase; gains during fine-tuning may reflect copy-rate rather than truth.
- **[MEDIUM]** **`side_by_side` has no comparator wired up.** `ARCHITECTURE.md:62-67` says "currently Qwen2.5-7B; Qwen2.5-72B GGUF Q4 once it's installed." The repo does not include a 72B comparator config; `artifacts/evals/base-qwen2.5-7b/` has no `side_by_side.json`. _Why it matters:_ the headline claim ("approaches 72B on domain tasks") has no machinery to verify it yet.
- **[MEDIUM]** **Synthetic-data prompts insert raw chunk text without any robustness mitigation.** `src/civic_slm/synth/generate.py:79-88` interpolates `chunk.text` directly into the templated prompt. _Why it matters:_ a single adversarially-worded staff report could derail synth labels. Even if Claude is robust in practice, the corpus has no sanitization or post-hoc validation that prompts followed instructions vs. doc content.
- **[MEDIUM]** **Hyperparameters in `configs/{cpt,sft,dpo}.yaml` are presented without rationale.** No comments explain the choice of LoRA r=64/α=128 (CPT) vs. r=32/α=64 (SFT/DPO), LR 1e-5 / 2e-4 / 5e-7, or the 2000/3-epoch/1-epoch budget. `ARCHITECTURE.md` stops at the contract level. _Why it matters:_ reviewers will ask, and "matches CLAUDE.md" is not a citation.
- **[INFO]** Baseline numbers reported in CLAUDE.md (factuality 0.501, refusal 0.800, extraction 0.277) reconcile exactly with `artifacts/evals/base-qwen2.5-7b/*.json`. The audit trail there is intact.

## 4. Security Engineer

- **[BLOCKER]** **No `SECURITY.md` and no documented vulnerability-disclosure path.** GitHub will surface a "Add a security policy" warning on day one. _Why it matters:_ for an ML project that pulls third-party model weights, scrapes external sites, and embeds an Anthropic API key, "email the maintainer" is not enough.
- **[HIGH]** **Synthetic-data pipeline is a prompt-injection sink.** `src/civic_slm/synth/generate.py:79-88` formats untrusted civic-document text into a templated prompt with no escaping or post-hoc validation. _Why it matters:_ a malicious or compromised civic site can inject instructions that pollute the SFT corpus (the very corpus that trains the model). At minimum this needs a documented threat-model entry; at maximum it's a poisoning vector with no detection.
- **[HIGH]** **Browser-driven crawler follows arbitrary redirects through `browser-use`.** `src/civic_slm/ingest/recipes/_browser.py` (the recipe runner) executes an LLM-driven agent against external sites with no domain-allowlist, no SSRF guard, no max-redirect cap visible at the recipe layer. _Why it matters:_ the agent can be talked into exfiltrating local resources or hammering a third-party service from the developer's machine. Civic sites are not adversarial today; this changes when the project is open-sourced and forked.
- **[HIGH]** **No SBOM and no pinned-hash supply-chain artifact.** `pyproject.toml:9-…` uses `>=` constraints; `uv.lock` exists but no SBOM (CycloneDX/SPDX) is produced or attested. _Why it matters:_ enterprise users and grant-funded civic-tech orgs increasingly require an SBOM for OSS adoption. The project is one `pip-audit`/`uv lock --upgrade` away from a transitive CVE escaping into someone's deployment.
- **[MEDIUM]** **`ignore-scripts`/postinstall posture is not documented.** Global env enforces it (per `~/.claude/CLAUDE.md`), but the repo doesn't tell forkers. _Why it matters:_ a contributor adding a dep with a `postinstall` script will silently miss our supply-chain hygiene.
- **[MEDIUM]** **HF model-download path is implicit.** `docs/RUNTIMES.md:206-209` explicitly notes strict-local does NOT prevent HF downloads. _Why it matters:_ that's correct behavior, but it's also the strongest path to model-weight tampering. No checksum verification of base weights is documented.
- **[INFO]** Secrets handling is good: read from `~/.config/civic-slm/.env` (`config.py:39-76`), never logged, missing-secret error is actionable (`config.py:74`). `lxml` CVE pin is explicit (`pyproject.toml:25-28, 63-69`). Strict-local tripwire correctly raises in both `llm/backend.py:120-128` and `ingest/recipes/_browser.py:50-58`. No secrets present in committed files (verified).

## 5. Privacy / Legal Counsel

- **[BLOCKER]** **No data-license audit on civic source corpora.** Nothing in the repo (no `DATA_CARD.md`, no `docs/LICENSE_DATA.md`, no entry in `docs/RECIPES.md` lines 1-171) records the terms of use of any specific city site. The implicit assumption is "publicly accessible website = redistributable training data," which is not the legal default in the U.S. — California cities each have separate site terms, and many ban automated scraping or commercial reuse. _Why it matters:_ this is the single most likely cease-and-desist vector at launch. CLAUDE.md (the audit prompt) explicitly designates this a BLOCKER absent a positively-confirmed license per source.
- **[BLOCKER]** **No `MODEL_CARD.md` and no `DATA_CARD.md`.** HF Hub expects both for any model+dataset push. `README.md:5` and `docs/USAGE.md:296-304` describe HF Hub as the release target. _Why it matters:_ shipping weights to HF without a model card with intended-use, known-failure-modes, demographic/geographic bias, and a "do not use for" section is both reputationally unsafe and against HF's own published expectations.
- **[BLOCKER]** **No PII-handling policy for public-meeting transcripts.** `src/civic_slm/ingest/video/caption.py:131-186` extracts `Speaker: text` lines from VTT/SRT, including the speaker label, and passes them through to `data/processed/`. There is no name-redaction, no address scrubbing, no opt-out flag. _Why it matters:_ public-comment periods at U.S. council meetings routinely include residents stating their full name and home address on the record because they were promised a local audience, not a global LLM training corpus. Brown Act / Public Records Act make the recording public — they do **not** authorize derivative LLM training. This is the embarrassment-at-launch finding most likely to make the front page.
- **[HIGH]** **No `LICENSE_WEIGHTS` / `LICENSE_DATA` separation.** `LICENSE` is MIT (covers code). The repo conflates code/weights/data licenses — `pyproject.toml:8` even disagrees with `LICENSE:1` (Apache-2.0 vs. MIT). Released model weights need their own license (OpenRAIL-M, Apache-2.0, or similar) and the dataset needs a third (CC-BY, ODC-BY, or "all rights reserved, non-redistributable"). _Why it matters:_ "MIT" on a fine-tuned Qwen2.5 derivative is not even necessarily compatible with Qwen's upstream license — that itself needs a finding.
- **[HIGH]** **No `ACCEPTABLE_USE_POLICY.md`.** `README.md` markets the model for "civic transparency tools." _Why it matters:_ there is no policy preventing someone from advertising the model as "civic legal advice," "voter eligibility assistant," or worse. A one-page AUP is table stakes.
- **[HIGH]** **No upstream-license check on the base model.** `configs/cpt.yaml` and `sft.yaml` reference `Qwen/Qwen2.5-7B-Instruct` as the base. The Qwen license has commercial-use thresholds and acceptable-use clauses. _Why it matters:_ if the released artifact violates Qwen's terms, every downstream user inherits the violation. Document it.
- **[MEDIUM]** **Synthetic data inherits the licensing of its seed corpus.** `synth/generate.py:79-88` produces examples whose factual content is taken verbatim from `chunk.text`. _Why it matters:_ the synthetic dataset is a derivative work of the source documents; Claude-as-author does not break the chain.
- **[MEDIUM]** **Trademark posture is undefined.** "civic-slm" itself is fine, but `README.md` references city names (San Clemente). _Why it matters:_ many U.S. cities trademark their seal. Add a one-line "we are not affiliated with…" disclaimer.
- **[INFO]** No real PII detected in `data/eval/*.jsonl` — all names, addresses, and entities are clearly synthetic (Coastal Surf School, ClearStreet Services, El Camino Real). Eval data is safe to ship as-is. The risk is what enters `data/processed/` from real crawls/transcripts, which the pipeline does not yet exist for end-to-end.

## 6. Product Manager

- **[HIGH]** **The whole "training pipeline" claim is unproven; only infrastructure exists.** `data/raw/`, `data/processed/`, `data/sft/`, and `data/dpo/` contain only `.gitkeep`. There is no v0 model artifact, no fine-tune run logged, no `artifacts/qwen-civic-*/`. _Why it matters:_ the project's value prop is "we beat base Qwen on civic tasks." Until even one fine-tune run completes, the README sells a future product. State this honestly on line 1.
- **[HIGH]** **No "first 5 minutes" experience.** `docs/USAGE.md` is a 349-line full pipeline; there is no `examples/` directory and no copy-paste 30-second "ask the model a question" demo. _Why it matters:_ the target persona (Code-for-America brigade, journalist) bounces.
- **[MEDIUM]** **The value-vs-RAG question is not answered anywhere.** Why fine-tune at all instead of base Qwen + RAG over the same corpus? `README.md` and `ARCHITECTURE.md` don't address it. _Why it matters:_ this is the first question every reviewer will ask, and silence reads as not-having-an-answer.
- **[MEDIUM]** **Success metrics for the project itself are absent.** `CLAUDE.md` lists eval bars; the README does not. _Why it matters:_ users can't tell if v0.1.0 is "ready" or "in progress."
- **[LOW]** **Scope discipline is mostly clean.** `ARCHITECTURE.md:169-175` explicitly out-of-scopes RAG, UI, multi-machine. Good — keep it that way.

## 7. Designer / Developer Experience

- **[MEDIUM]** **README opens with implementation, not problem.** `README.md:1-5` leads with "fine-tune of Qwen2.5-7B" before answering "why does a civic technologist care?" _Why it matters:_ the audience is two-thirds non-ML.
- **[MEDIUM]** **No examples directory, no example notebook, no demo prompt.** `docs/USAGE.md` is the only entry point and is procedural. _Why it matters:_ the "show me what it does" path doesn't exist.
- **[LOW]** **Quickstart secret-path is not idempotent.** `README.md:31` and `docs/USAGE.md:11` direct users to create `~/.config/civic-slm/.env` manually with no script or `civic-slm doctor --setup`. `civic-slm doctor` (`src/civic_slm/doctor.py`) checks but does not bootstrap. _Why it matters:_ small papercut on a first-five-minutes flow.
- **[INFO]** CLI ergonomics are strong: Typer subcommands with help text on every option, actionable error messages (`config.py:74`, `train/dpo.py:66-71`, `ingest/video/asr.py:36-39`), structlog throughout. CHANGELOG follows Keep-a-Changelog. CONTRIBUTING.md is well-scoped.

## 8. DevRel / Community

- **[BLOCKER]** **No `CODE_OF_CONDUCT.md`.** Required by GitHub community standards and by every reasonable contributor. _Why it matters:_ for a project explicitly courting civic-tech volunteers, a CoC is non-negotiable.
- **[HIGH]** **No `.github/ISSUE_TEMPLATE/` and no `PULL_REQUEST_TEMPLATE.md`.** Verified absent (only `.github/workflows/ci.yml` exists). _Why it matters:_ contributors don't know what info you need; you'll spend the first month asking for repro steps.
- **[HIGH]** **No governance, no roadmap, no maintainer list.** `CONTRIBUTING.md` documents _how_ to contribute but not _who decides_ or _what's coming_. _Why it matters:_ "what happens when 50 cities ask for their data added?" has no answer.
- **[MEDIUM]** **No labels/triage scheme defined.** No "good first issue" plan; no SLA on PR review. _Why it matters:_ small thing, but the difference between a healthy and a moribund OSS project.
- **[MEDIUM]** **No discussion forum / community channel.** `README.md:112` lists only doc files. _Why it matters:_ users have nowhere to go that isn't an issue.

## 9. Technical Writer / Documentation

- **[HIGH]** **No model card and no data card.** Diátaxis-wise, the project has _tutorial_ (USAGE), _how-to_ (RECIPES, RUNTIMES), and _explanation_ (ARCHITECTURE) — but no _reference_ artifact for the model and dataset themselves. _Why it matters:_ this is the standard-of-the-field for ML releases (HF, Google PAIR, BigScience).
- **[MEDIUM]** **Doc-code drift between top-of-package docstring and README.** `src/civic_slm/__init__.py:1` says "CA municipal documents"; `README.md:3` says "U.S. local-government across all 50 states." _Why it matters:_ the kind of small drift that compounds.
- **[MEDIUM]** **No glossary for non-ML civic readers.** `LoRA`, `DPO`, `CPT`, `LoRA r/α`, `quantize`, `chat template` appear without definition. _Why it matters:_ target persona explicitly includes journalists and brigade volunteers.
- **[MEDIUM]** **No architecture diagram.** `ARCHITECTURE.md` is text-only over 175 lines. _Why it matters:_ a one-page block diagram saves an hour for every reader.
- **[LOW]** **CHANGELOG is well-disciplined** (`CHANGELOG.md` follows Keep-a-Changelog, line 3). One nit: v0.1.0 is dated `2026-04-24`; today is `2026-04-25` — fine, but if you re-tag, sync the date.

## 10. Open Source Strategist

- **[BLOCKER]** **License ambiguity (see Principal Engineer #1).** Until pyproject and LICENSE agree, the project is not actually open-source-licensable in any defensible way.
- **[HIGH]** **Release depends on a paid API in the critical path.** `synth/generate.py` uses Claude via `llm/backend.py`. `docs/RUNTIMES.md:56-62` is honest that fully-local synth is possible, but the **default** path (`CIVIC_SLM_LLM_BACKEND` defaults to `anthropic` per `llm/backend.py:120-128`) requires API spend. _Why it matters:_ for many forkers (universities, civic non-profits), Anthropic credits are a hard fork-blocker. State this on README line 2 and consider flipping the default.
- **[HIGH]** **Commercial-use stance is undocumented.** No statement in README, LICENSE-set, or AUP covers commercial reuse of weights. _Why it matters:_ grant-funded civic tech needs to know.
- **[MEDIUM]** **No CLA / DCO chosen.** `CONTRIBUTING.md` is silent on this. _Why it matters:_ a future relicensing / corporate stewardship transition is easier with one in place from day one.
- **[MEDIUM]** **No naming/branding clearance noted.** "civic-slm" is generic; this is fine. But CLI command `civic-slm` and the HF org name need to be confirmed available before tagging v1.0.

## 11. Civic Technologist (target user persona)

- **[HIGH]** **The project is California-shaped under a 50-state label.** The only registered recipe is `san-clemente` (`src/civic_slm/cli.py:26`, `ingest/recipes/san_clemente.py`). All eval contexts are CA-shaped (El Camino Real, CUP nomenclature). The package docstring still says "CA municipal documents" (`__init__.py:1`). _Why it matters:_ a Texas brigade volunteer who follows the README will write a recipe, find no other examples, and bounce. The "all 50 states" framing is currently aspirational.
- **[MEDIUM]** **California-isms presented as universal.** `recipes/san_clemente.py` and the eval contexts use CEQA-flavored language and CUP/staff-report formats. Texas, New York, and Ohio do these very differently. `docs/RECIPES.md` does not warn about this. _Why it matters:_ silently encourages monoculture.
- **[MEDIUM]** **Hardware bar is high but documented honestly.** `docs/RUNTIMES.md:50-62` says ≥32GB unified memory recommended; `CONTRIBUTING.md:6-10` says Apple-Silicon-only for training. _Why it matters:_ honest, but excludes most volunteers. Document a "what works on a 16GB M1 MacBook Air" path or say it doesn't.
- **[MEDIUM]** **`browser-use` + LLM-driven crawling needs an Anthropic key by default.** A volunteer journalist with no API budget cannot crawl. _Why it matters:_ the tool stops being self-serve at the first step. Strict-local plus the local backend works, but takes a separate doc.

## 12. Release Engineer

- **[BLOCKER]** **No release reproducibility from VERSION → built artifact.** `pyproject.toml:3` and `VERSION` disagree. There is no `dynamic = ["version"]` wiring, no `__version__` derivation, no `civic-slm version` CLI command. _Why it matters:_ a tagged v0.1.0 release will publish a `0.0.0` package.
- **[BLOCKER]** **No `SECURITY.md`** (also Security Engineer #1). _Why it matters:_ release-checklist standard.
- **[HIGH]** **No release checklist.** `CONTRIBUTING.md:125-127` mentions changelog + version-bump discipline, but there is no `RELEASING.md` covering: bump VERSION, bump pyproject, sync `__init__.__version__`, regenerate baselines, tag, push to HF, push to PyPI. _Why it matters:_ first solo release will skip a step.
- **[HIGH]** **No artifact signing / provenance.** No sigstore, no `attestations.yml`, no model-weight signing. _Why it matters:_ standard for OSS ML releases in 2026.
- **[MEDIUM]** **No `CITATION.cff`.** Researchers will need to cite. _Why it matters:_ small file, large signal.
- **[MEDIUM]** **No deprecation policy.** "v1.0" commits to nothing about backward-compatibility. _Why it matters:_ future-you will regret this in 6 months.
- **[INFO]** CI is sound: `.github/workflows/ci.yml:19-57` runs ruff lint + format check + pyright strict + pytest on a 2×2 matrix. `lxml` CVE is pinned with explanatory comment (`pyproject.toml:25-28, 63-69`). Test count is 65, fast (~0.15s).

---

## Wildcard

- **[BLOCKER — Wildcard]** **The combined risk of (a) public-meeting PII passing through the transcript pipeline, (b) no data-license audit, and (c) a degenerate refusal benchmark is reputationally larger than any single finding.** A reasonable reporter could open this repo today, find a refusal benchmark that gives credit to a "refuse everything" model, an ingest path that vacuums up resident addresses from city YouTube streams, and a README that promises "all 50 states" while shipping one CA recipe — and write a story. The rest of the repo is genuinely strong; that contrast makes the negative findings sting more, not less.

---

## Executive summary

### Top 5 BLOCKERs

1. **License declaration is internally inconsistent** — `pyproject.toml:8` (Apache-2.0) vs. `LICENSE:1` (MIT). [§1, §10, §12]
2. **Refusal benchmark is degenerate** — 10/10 examples expect refusal; a constant-refuse model scores 1.0. The safety story is broken until a should-answer negative class is added. [§3]
3. **No source-document hash on eval examples; no train/eval contamination check anywhere in the pipeline.** [§3]
4. **No `MODEL_CARD.md` / `DATA_CARD.md` and no data-license audit on civic source corpora.** [§5, §9]
5. **No PII-handling policy for public-meeting transcripts** — speaker labels (and any volunteered names/addresses) flow through unredacted. [§5]

### Top 5 HIGHs

1. **`pyproject.toml` version is `0.0.0`**, contradicting `VERSION` (`0.1.0`) and the v0.1.0 CHANGELOG entry. [§1, §12]
2. **`pyproject.toml` description ("California municipal documents") and `__init__.py:1` docstring are stale**, contradicting the all-50-states framing. [§1, §11]
3. **The training pipeline has shipped no model artifact** — `data/raw/`, `data/sft/`, `data/dpo/` are empty; the headline value-prop is unproven. [§6]
4. **Eval n=5–10 with no seeds, no temperature, no CIs, no second-city held-out** — no defensible quantitative claim is possible yet. [§3]
5. **No `CODE_OF_CONDUCT.md`, no `SECURITY.md`, no issue/PR templates, no governance** — community-readiness gaps. [§4, §8, §12]

### Release-readiness scores (1–10)

| Stakeholder        | Score | One-line justification                                                                                         |
| ------------------ | ----- | -------------------------------------------------------------------------------------------------------------- |
| Principal Engineer | 6     | Strict pyright, clean tests, no TODOs — undermined by license/version-metadata mismatch.                       |
| SRE / Infra        | 6     | Crawl is content-addressed and idempotent; training has no resume and silently swallows W&B failures.          |
| ML Research        | 3     | Refusal benchmark is degenerate; no contamination check; n=5–10 with no seeds. The most exposed dimension.     |
| Security           | 4     | Excellent secret hygiene + strict-local tripwire, but no SECURITY.md, no SBOM, prompt-injection sink in synth. |
| Privacy / Legal    | 2     | No data-license audit, no model/data card, no transcript PII policy. The biggest legal exposure.               |
| Product            | 5     | Clear scope and out-of-scope discipline, but no shipped model and no first-5-minutes story.                    |
| Designer / DevEx   | 7     | Strong CLI ergonomics and CONTRIBUTING; missing examples and a problem-first README.                           |
| DevRel / Community | 3     | Missing CoC, templates, governance, roadmap, channels.                                                         |
| Technical Writer   | 6     | USAGE/RECIPES/RUNTIMES are good; missing model card, data card, glossary, diagram.                             |
| OSS Strategist     | 4     | License ambiguity + Anthropic-paid default path are significant fork-blockers.                                 |
| Civic Technologist | 4     | CA-shaped under a 50-state label, hardware-restricted, paid-API default.                                       |
| Release Engineer   | 3     | Version metadata broken, no release checklist, no signing, no SECURITY.md.                                     |

**Composite read:** the engineering substrate is genuinely strong (clean code, strict typing, fast tests, careful secrets, real CVE awareness). The release-readiness gaps are governance, legal, and ML-rigor — exactly the layers a reasonable reviewer will land on first.

---

## Recommended cut line

**Minimum required to open-source responsibly (must-do before any public tag):**

1. Reconcile licenses across `LICENSE`, `pyproject.toml:8`, package docstring, README, and any future weights/data card. Pick three licenses (code/weights/data) explicitly. [BLOCKER §1, §5, §10]
2. Sync `pyproject.toml:3-4` with `VERSION` and the README description; wire `__version__` into `__init__.py:1` so it isn't stale. [BLOCKER §1, §12]
3. Land `SECURITY.md`, `CODE_OF_CONDUCT.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `ACCEPTABLE_USE_POLICY.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`. The model card must include known-failure-modes, geographic bias (CA-only), and a do-not-use-for section. [BLOCKER §4, §5, §8, §9]
4. Add a should-answer negative class to `data/eval/refusal.jsonl` (target ≥30% of records, 30+ records) and re-baseline. Until done, do not publish refusal numbers in README/blog/HF card. [BLOCKER §3]
5. Add `source_doc_hash` to `Provenance` (`schema.py:113-120`) and to all `EvalExample` subclasses; add a runtime check in `eval/runner.py` that raises if any eval source hash appears in the train manifest. Even with empty corpora today, the schema must be in place before the first real ingest. [BLOCKER §3]
6. Write the data-license audit per source. For each registered recipe, document the source-site terms-of-use and the legal basis for redistribution. [BLOCKER §5]
7. Add transcript-PII handling to `ingest/video/caption.py` (at minimum: an opt-in flag that strips speaker labels from public-comment cues; documented policy on resident-volunteered names/addresses). [BLOCKER §5]

**What can wait to v1 / next quarter:**

- BGE-reranker swap of factuality scorer (HIGH §3) — current scorer is honest in the architecture doc; document the limitation in the model card and ship.
- Eval scale-up to 200/100/50/100 (HIGH §3).
- 72B comparator wiring + side-by-side numbers (MEDIUM §3).
- SBOM/sigstore artifact signing (HIGH §4, §12).
- CITATION.cff, governance/roadmap, discussion channel (MEDIUM §8).
- First-5-minutes example notebook + value-vs-RAG paragraph (MEDIUM §6, §7).
- Architecture diagram + glossary (MEDIUM §9).
- Second-city held-out eval (HIGH §3) — required before any "all-50-states" headline claim, but not required to tag v0.1.0 as "infrastructure preview."

**Pragmatic framing:** v0.1.0 is best published as **"infrastructure preview, no shipped model yet"** — that lets the engineering quality carry the launch and turns the empty `data/sft/` and unproven training claims into honest signposting rather than oversell. Tag as `0.1.0-pre` or `0.1.0` with a top-of-README banner; v1.0 happens after the cut-line items above are closed and at least one fine-tune actually beats the baselines on the four benchmarks.
