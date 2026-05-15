"""Release-readiness pipeline smoke (#55).

Two cheap gates that catch class-of-error bugs the unit tests miss:

1. `civic-slm train jurisdiction <slug> --dry-run` resolves for every
   registered recipe. The composer can fail at slug lookup, config
   template discovery, or env wiring — `--dry-run` exercises all three
   without touching the network or training. The unit tests for
   `_load_recipes()` don't catch this because they don't go through the
   composer.

2. `civic-slm doctor` runs to completion without crashing when no LM
   Studio is reachable. CI doesn't have an LM Studio; doctor *should*
   exit non-zero (the candidate ping fails), but it should do so with a
   `typer.Exit`, not an uncaught exception.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from civic_slm.cli import app as civic_slm_app
from civic_slm.ingest.crawl import _load_recipes  # pyright: ignore[reportPrivateUsage]

runner = CliRunner()


def _recipe_slugs() -> list[str]:
    return sorted(_load_recipes().keys())


@pytest.mark.parametrize("slug", _recipe_slugs())
def test_train_jurisdiction_dry_run_resolves_every_recipe(slug: str) -> None:
    """Composer must resolve every shipped slug. `--dry-run` exits before
    crawl/synth/train so this stays a unit test.

    Failure modes this catches: typo in a slug, missing config template
    at `configs/jurisdiction-default.{cpt,sft}.yaml`, broken import in
    `civic_slm.train.jurisdiction`, env wiring drift in `configure()`.
    """
    result = runner.invoke(
        civic_slm_app,
        ["train", "jurisdiction", slug, "--dry-run"],
    )
    assert result.exit_code == 0, (
        f"`civic-slm train jurisdiction {slug} --dry-run` exited "
        f"{result.exit_code}:\n{result.output}\n{result.exception!r}"
    )
    # Sanity: the plan output mentions the slug somewhere.
    assert slug in result.output, f"dry-run output didn't mention {slug!r}:\n{result.output}"


def test_doctor_does_not_crash_without_lm_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI has no LM Studio. Doctor should exit with `typer.Exit(1)` after
    the candidate ping fails — not an uncaught exception.

    Verifies: command imports cleanly, fail-mode is the well-defined one
    (`typer.Exit` from the `overall == "fail"` branch), no NameErrors or
    Pydantic validation surprises that future env-var drift could introduce.
    """
    # Make sure we hit the failure path deterministically — point the
    # candidate URL at a TCP port nobody is listening on.
    monkeypatch.setenv("CIVIC_SLM_LM_STUDIO_URL", "http://127.0.0.1:1")
    # Clear secrets so the secret-check branches all run.
    for key in ("ANTHROPIC_API_KEY", "HF_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "local")

    from civic_slm import doctor

    # `doctor.main` is a Typer command; invoke it via CliRunner so
    # `typer.Exit` is captured as an exit code instead of raising.
    result = runner.invoke(doctor.app, [])

    # Allow exit code 0 (everything happened to work) or 1 (the expected
    # ping-failed path). Anything else (2, segfault, uncaught exception)
    # is the failure mode this test guards.
    assert result.exit_code in (0, 1), (
        f"doctor exited unexpectedly: code={result.exit_code}\n"
        f"output:\n{result.output}\nexception: {result.exception!r}"
    )
    # `typer.Exit(1)` surfaces through CliRunner as `SystemExit(1)`. Any other
    # exception type would be an uncaught crash — that's the regression we're
    # guarding against.
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise AssertionError(
            f"doctor raised an uncaught exception (not SystemExit/typer.Exit): "
            f"{type(result.exception).__name__}: {result.exception}\n"
            f"output:\n{result.output}"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
