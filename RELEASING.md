# Releasing

This is the maintainer's checklist for cutting a release. The `VERSION` file
is the source of truth; everything else either reads from it
(`pyproject.toml` via `tool.hatch.version`) or must be re-synced.

## When to bump

`civic-slm` follows [SemVer](https://semver.org/) with one project-specific
clarification:

| Change                                                 | Version bump                                                               |
| ------------------------------------------------------ | -------------------------------------------------------------------------- |
| New recipe, new doc, new optional CLI flag             | patch                                                                      |
| New eval bench, new public function, new env var       | minor                                                                      |
| Removed CLI flag, renamed module, schema field removal | major (or pre-1.0: minor with a CHANGELOG `Breaking` block)                |
| New shipped model artifact (HF Hub release)            | minor at minimum (and the model card lists actual measured numbers)        |
| Eval-harness change that moves baselines               | minor + re-baseline + CHANGELOG note that prior numbers are not comparable |

Pre-1.0, breaking changes are allowed in minor versions but **must** appear
in `CHANGELOG.md` under a `### Breaking` heading inside the version section.

## Pre-flight checklist

### Cross-process signal smoke test (manual)

The supervisor's signal-forwarding path is not load-bearing in CI
(coordinating cross-process SIGINTs in pytest is flaky on macOS). Verify
manually before tagging any release that ships a real training run:

```bash
# Start a smoke CPT in one terminal:
uv run civic-slm train cpt --smoke-test
# In another terminal, find the supervisor pid (`ps`) and SIGINT it.
# Confirm: the child exits, an adapter file lands in artifacts/qwen-civic-cpt/,
# and re-running without --resume aborts with the resume-guard message.
```

```bash
# 1. Up-to-date main, clean tree.
git checkout main
git pull
git status -s   # should be empty

# 2. Local quality gates pass.
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q

# 3. Doctor passes (and strict-local passes if you intend to advertise it).
uv run civic-slm doctor
CIVIC_SLM_LLM_BACKEND=local CIVIC_SLM_STRICT_LOCAL=1 \
    uv run civic-slm doctor --strict-local

# 4. Eval baselines are still the published numbers (regressions in the
#    scorer or runner can mask real model gains — re-confirm before
#    tagging anything that touches eval/).
uv run pytest tests/test_scorers.py tests/test_eval_runner.py -q

# 5. Release-readiness gates pass (see "CI gates" below for what these
#    enforce mechanically).
uv run pytest tests/test_release_gates.py tests/test_pipeline_smoke.py -q
uv lock --check
```

## CI gates (mechanical invariants)

CI's `lint-type-test` job enforces the project-level invariants below in
addition to ruff / pyright / pytest. Anything not on this list is still
maintainer responsibility — these are the ones we've moved from tribal
knowledge to `git rejects the PR` (see #55).

- **Every YAML recipe parses** (`tests/test_release_gates.py::test_every_yaml_recipe_parses`).
- **Every recipe has a `docs/SOURCES.md` audit row, and every audit row points at a real recipe.** Catches leftover entries after a rename.
- **Every `Decision:` line resolves to GO / NO-GO / PENDING.** Catches case typos.
- **No `*.bin` / `*.pdf` / `*.safetensors` / `*.gguf` / audio committed under `data/raw/`.** Belt-and-braces beyond `.gitignore`.
- **No SSN / credit-card-shape / DOB / personal-domain email in committed `data/eval/*.jsonl` or `data/sft/*.jsonl`.** Public-record contacts (`.gov` emails, council phone numbers, permit addresses) are allowed by design.
- **Every `data/eval/*.jsonl` line round-trips through the `EvalExample` discriminated union** (`tests/test_eval_runner.py::test_every_committed_bench_jsonl_validates`).
- **Every `artifacts/evals/<model>/<bench>.json` row parses as `EvalResult`.** Catches runner-output drift.
- **MODEL_CARD numbers match `artifacts/evals/` means to 4 decimals** (`tests/test_eval_runner.py::test_model_card_numbers_match_artifacts`). The v1-blocker invariant.
- **`civic-slm train jurisdiction <slug> --dry-run` resolves for every shipped recipe** (`tests/test_pipeline_smoke.py`).
- **`civic-slm doctor` exits with `typer.Exit`, not an uncaught exception, when LM Studio isn't reachable.**
- **`uv.lock` is in sync with `pyproject.toml`** (CI step `Lockfile in sync`).
- **Markdown links resolve, including `#anchor` fragments** (CI job `docs link check`, lychee).

If you need to bypass one of these for a legitimate reason (e.g. a bench
file is intentionally being held under-validated during a migration),
mark it `pytest.skip` with a comment naming the issue tracking the
follow-up. Don't silently delete the test.

## Branch protection

`scripts/apply-branch-protection.sh` is the one-shot script that sets
GitHub's `main`-branch policy to match what CI enforces. Required checks,
linear history, no force-push, no branch deletion.

```bash
# Run once, after any change to the matrix in .github/workflows/ci.yml
# that adds or removes a required check.
./scripts/apply-branch-protection.sh

# Override the repo target (default itsmeduncan/civic-slm):
OWNER=someone REPO=fork ./scripts/apply-branch-protection.sh

# Inspect the live policy:
gh api repos/itsmeduncan/civic-slm/branches/main/protection \
  --jq '.required_status_checks.contexts'
```

The maintainer's `--admin` flag on `gh pr merge` bypasses these checks
for unblocking (e.g. an external CI outage). Use sparingly.

## Cut the release

```bash
# 1. Bump VERSION to the new value (e.g., 0.2.0).
echo "0.2.0" > VERSION

# 2. Move the [Unreleased] section in CHANGELOG.md under a new
#    `## [0.2.0] - YYYY-MM-DD` heading. Leave [Unreleased] empty above it.

# 3. Verify the version pipeline.
uv sync --quiet
uv run python -c "import civic_slm; print(civic_slm.__version__)"
# → 0.2.0

# 4. Commit, tag, push.
git add VERSION CHANGELOG.md
git commit -m "chore: release v0.2.0"
git tag -a v0.2.0 -m "civic-slm v0.2.0"
git push origin main --tags
```

## Post-tag

- Open a GitHub Release pointing at the tag; paste the relevant
  `CHANGELOG.md` section as the body.
- If the release ships a new model artifact:
  1. Run all four eval benches against the new model and commit the
     results to `artifacts/evals/qwen-civic-vN/`.
  2. Write `artifacts/qwen-civic-vN/MODEL_CARD.md` with the actual
     measured numbers (the top-level `MODEL_CARD.md` is the contract;
     the per-version one is the receipt).
  3. **Sign the artifacts**: `scripts/sign-release.sh artifacts/civic-slm-vN-mlx-q4 artifacts/civic-slm-vN-gguf-q5km`. See [Signing released artifacts](#signing-released-artifacts) above for the full flow.
  4. Push to HF Hub: `mlx-community/civic-slm-vN` and
     `civic-slm/civic-slm-vN-gguf`. Include `sha256sums.txt` + `sha256sums.txt.bundle` in the upload so downstream consumers can verify.
  5. Attach the same two signature files to the GitHub Release.
  6. Update the top-level `MODEL_CARD.md` "civic-slm v1 target" column
     with measured numbers in a new "Measured (v1)" column. Do **not**
     overwrite targets — the next planned release wants its own column.

## Yanking a release

Releases are not yanked from PyPI / HF — that is too disruptive for
downstream forks. Instead, ship a hotfix patch release and document the
issue in `CHANGELOG.md` under the affected version's `### Notes` block.

For genuinely unsafe releases (e.g., the model leaks PII), a yank may be
warranted. Coordinate with affected forks first; coordinate disclosure
with `SECURITY.md`.

## Deprecation policy

Pre-1.0:

- Anything not under `civic_slm.<module>.<public>` may change without
  notice. Code marked `# pyright: ignore[reportPrivateUsage]` is fair
  game.
- A removed public function or CLI flag must appear in `CHANGELOG.md`
  under `### Breaking` in the same release that removes it. There is no
  guaranteed soft-deprecation window pre-1.0, but maintainers will
  add one (`warnings.warn(DeprecationWarning, ...)` for ≥1 minor) when
  it's cheap.

Post-1.0 (planned, not yet binding):

- Public APIs (anything documented in `README.md`, `docs/`, or
  exposed by `civic-slm <subcommand>`) carry a one-minor-version
  deprecation window. Removals require a `DeprecationWarning` in the
  prior minor release pointing at the replacement.
- Schema changes to artifacts in `data/` (e.g., `Provenance`,
  `EvalExample`) are breaking. Migration tooling ships in the same
  release that introduces the change.
- Eval benchmarks are versioned. Adding examples is non-breaking;
  changing scoring is a major bump unless the change is documented to
  produce strictly higher scores on the same population.

## Signing released artifacts

Released model artifacts (MLX-q4 + GGUF Q5_K_M) are signed with Sigstore
keyless OIDC. The signed thing is a `sha256sums.txt` manifest covering
every file in the artifact directory; one signature covers the whole
release. Verification re-hashes the files, compares against the
manifest, and checks the cosign bundle's certificate identity. Closes
#27.

### Canonical release identity

These two values are pinned in `scripts/verify-release.sh` and must
match the OAuth flow you complete at sign time. **Don't rotate without
updating both scripts AND this document in the same commit** — the
identity test in `tests/test_signing.py` will fail loudly if these
drift.

| Field                     | Value                            |
| ------------------------- | -------------------------------- |
| Identity (cert SAN email) | `itsmeduncan@gmail.com`          |
| OIDC issuer               | `https://github.com/login/oauth` |

Forks signing their own releases should override:

```bash
CIVIC_SLM_RELEASE_IDENTITY=alice@example.org \
CIVIC_SLM_RELEASE_OIDC_ISSUER=https://accounts.google.com \
    scripts/sign-release.sh artifacts/your-model-dir
```

### Signing (maintainer, post-fuse)

```bash
# 1. Install cosign once.
brew install cosign   # macOS
# or: see https://docs.sigstore.dev/system_config/installation/

# 2. After `civic-slm merge --version vN` produces the artifact dirs,
#    sign each one. A browser opens for OAuth — sign in as the canonical
#    GitHub identity (itsmeduncan@gmail.com).
scripts/sign-release.sh \
    artifacts/civic-slm-vN-mlx-q4 \
    artifacts/civic-slm-vN-gguf-q5km

# 3. Upload both files alongside the model when pushing to HF Hub:
#       sha256sums.txt
#       sha256sums.txt.bundle
#    huggingface_hub.upload_folder picks them up automatically.

# 4. Attach the same two files to the GitHub Release for the tag.
```

### Verifying (downstream consumer)

```bash
# 1. Download the artifact directory from HF Hub or the GitHub Release.
#    Make sure sha256sums.txt and sha256sums.txt.bundle land in the
#    same directory as the model files.

# 2. Verify. Exits 0 only if every file matches the manifest AND the
#    manifest's signature was issued to the canonical identity.
scripts/verify-release.sh ./downloaded-model-dir
```

A passing verify means: the bytes on disk are exactly what the
maintainer signed, and the signer was the holder of the pinned identity
at sign time (Fulcio short-lived cert, Rekor transparency-log entry).
No long-lived public key in this repo to compromise — that's the
keyless-OIDC story.
