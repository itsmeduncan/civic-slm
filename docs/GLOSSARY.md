# Glossary

Plain-language definitions for terms that appear in this repo. Aimed at
civic technologists, journalists, and brigade volunteers who may not have
an ML background. Linked from `README.md` and `MODEL_CARD.md`.

## ML and training terms

**Adapter (LoRA adapter).** A small set of weights that modifies a frozen
base model. Instead of changing all 7 billion parameters of Qwen2.5-7B,
LoRA changes a few million parameters layered on top. Cheap to train,
cheap to ship, swappable.

**Base model.** The model we start from before any fine-tuning. Here, that
is `Qwen/Qwen2.5-7B-Instruct` — Alibaba's open-weights chat model.

**CPT (continued pretraining).** A short pass over raw text from the
target domain (civic documents) so the model learns the _vocabulary and
rhythm_ of those documents before any task-specific tuning happens.
Think of it as "soak" time.

**SFT (supervised fine-tuning).** Training on (input, output) pairs to
teach the model how to perform specific tasks — answer civic questions
with citations, extract structured data from staff reports, refuse when
the answer is not in context. Most of the model's task knowledge comes
from this stage.

**DPO (direct preference optimization).** Training on (chosen, rejected)
pairs to nudge the model toward the preferred behavior on a task without
re-teaching the task itself. Used here mostly for refusal calibration
and citation rigor.

**Eval / benchmark.** A held-out set of examples used to measure how the
model performs. We have four (`data/eval/*.jsonl`):

- _factuality_ — can it answer a grounded question with a correct citation?
- _refusal_ — does it decline when the answer is not in context (and
  answer when it is)?
- _extraction_ — can it produce a JSON object with the right fields
  given a staff report?
- _side-by-side_ — when judged head-to-head against a comparator, does
  it win?

**Hyperparameters.** Settings that control training: learning rate, LoRA
rank, number of epochs, batch size. See `configs/{cpt,sft,dpo}.yaml`,
which now ship with inline rationale for every value.

**LoRA rank.** A capacity dial for the adapter. Higher rank = more
parameters = more capacity to learn, but also more memory and more risk
of overfitting on a small corpus.

**Quantization (q4 / Q5_K_M).** Storing the weights at lower precision
(4 or 5 bits per weight instead of 16) to fit the model into less memory
on an end-user machine. We ship two quantized formats:

- **MLX 4-bit** — the primary Mac artifact, fastest on Apple Silicon.
- **GGUF Q5_K_M** — the broadly-compatible artifact for `llama.cpp` and
  Ollama; runs on Linux, Windows, and Macs.

**Tokens / tokenization.** Models read text in pieces called tokens
(roughly 4 characters or 0.75 words on average). Sequence length, batch
size, and cost are all measured in tokens.

**Train / inference template parity.** The chat format used at training
time must match the format the model sees at serving time. We rely on
Qwen's built-in tokenizer to apply both, so they stay in lock-step.

## Civic / government terms

**Brown Act.** California's open-meetings law (Gov. Code §54950 et seq.).
Requires local-government bodies to deliberate in public, publish
agendas in advance, and make minutes available. Texas, New York, and
other states have their own equivalents.

**CEQA.** California Environmental Quality Act. Requires environmental
review of public projects. Mentioned in many California staff reports
(e.g., as "exempt under CEQA §15061(b)(3)"). Texas and other states
have analogous but distinct review regimes.

**Comprehensive plan / general plan / master plan.** A jurisdiction's
long-range planning document. California uses "general plan," most
other states use "comprehensive plan" or "master plan" for the same
artifact. All three are accepted by `civic_slm.schema.DocType`.

**CUP (conditional use permit).** A permit that allows a specific use of
a property that is not permitted by-right under zoning. Heard by a
planning commission. Common in agendas and staff reports.

**Jurisdiction.** A unit of local government — a city, county,
township, or school district. The repo's `Recipe` Protocol takes
`jurisdiction` (slug, e.g. `san-clemente`) and `state` (USPS code,
e.g. `CA`) so a fork can target Texas or Ohio without code changes.

**Municipal code.** A jurisdiction's codified ordinances — the standing
rules covering zoning, business licensing, public works, etc. Often
hundreds of pages.

**Ordinance / resolution.** Individual legislative items. An ordinance
amends the municipal code; a resolution states a policy or
authorization.

**Public records statute.** The state law that makes government records
publicly inspectable. California Public Records Act (Gov. Code §7920+),
Texas Public Information Act (Gov. Code Ch. 552), New York Freedom of
Information Law, etc. Each `Recipe` lands in `docs/SOURCES.md` with the
specific statute that places its source documents in scope.

**Staff report.** A short document prepared by city/county staff for an
elected body, recommending action on a specific agenda item. Has a
predictable structure (subject, recommendation, background, fiscal
impact, environmental review). The `extraction` benchmark targets this
structure.

## Repo / tooling terms

**Recipe.** A small Python file in `src/civic_slm/ingest/recipes/` that
tells the crawler how to find documents for one jurisdiction. Recipes
follow the `Recipe` Protocol and ship with a license-audit entry in
`docs/SOURCES.md`.

**Manifest.** `data/raw/manifest.jsonl` — the append-only, content-
addressed journal of every document we have ingested. Committed to the
repo even when the raw bytes are gitignored, so the corpus is
auditable.

**Provenance.** `civic_slm.schema.Provenance` — the record of how a
synthetic example was generated: model id, prompt-template hash,
source-document hash, timestamp. Used by the eval contamination check
in `civic_slm.eval.runner.assert_no_contamination()`.

**Strict-local mode.** `CIVIC_SLM_STRICT_LOCAL=1` — forces synth, judge,
and crawler to refuse Anthropic at runtime. Pair with
`civic-slm doctor --strict-local` for a one-shot proof that no code
path can reach a paid endpoint. Useful for forks without an API budget
and for verified-zero-spend audits.

**W&B (Weights & Biases).** Optional training-run dashboard. Set
`WANDB_API_KEY` to enable; init failures are logged as warnings, not
errors, so missing W&B never aborts a training run.

**BGE similarity (factuality scorer).** `civic-slm eval run --similarity bge`
loads `BAAI/bge-large-en-v1.5` (a sentence-transformers dual encoder),
embeds the gold answer and the model prediction, takes cosine similarity,
and maps it to `[0, 1]`. The default scorer is still word-overlap (no
extra dependency, fast, but rewards verbatim copying); BGE is the opt-in
"semantic similarity" upgrade. Numbers under the two scorers don't compare
— the eval JSONL `_run_config` header records which one produced the run.

**Supervisor (training).** `src/civic_slm/train/supervisor.run_supervised`.
A thin wrapper around the `mlx_lm` subprocess that propagates `SIGTERM`
and `SIGINT` to the child so a Ctrl-C lets the trainer flush a checkpoint
cleanly. After 10s without exit, escalates to `SIGKILL`. Non-zero exits
raise `TrainerError` with the exit code in the message.

**Resume guard.** `civic-slm train cpt|sft|dpo` refuses to start when the
configured `output_dir` already contains a `.safetensors` adapter file,
unless you pass `--resume`. Prevents the previous footgun where re-running
the trainer silently overwrote a prior run.

**Smoke test (training).** `civic-slm train cpt|sft|dpo --smoke-test`. CPT
runs 100 iters, SFT/DPO 50. Skips the resume guard since smoke runs are
throwaway. Per CLAUDE.md: smoke before every real run.

**Comparator (side_by_side).** The 72B model that the candidate is judged
against in `civic-slm eval side-by-side`. Stood up as a separate
`llama-server` on port 8081 (default) hosting `Qwen2.5-72B-Instruct-Q4_K_M`.
The runner pings the comparator before any candidate work and raises
`ComparatorMissingError` if it isn't reachable. See `docs/RUNTIMES.md`
"Standing up the 72B comparator" for the runbook.
