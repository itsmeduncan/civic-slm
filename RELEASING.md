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
```

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
  3. Push to HF Hub: `mlx-community/civic-slm-vN` and
     `civic-slm/civic-slm-vN-gguf`.
  4. Update the top-level `MODEL_CARD.md` "civic-slm v1 target" column
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

## Signing (planned, not yet implemented)

Per `ROADMAP.md`, sigstore signing of git tags + HF artifacts lands in
v1.0. Until then, releases carry no signed provenance and downstream
forks should pin to commit SHAs rather than tags if integrity matters.
