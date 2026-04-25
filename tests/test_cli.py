"""CLI smoke tests.

Verify that every subcommand Typer claims to expose is actually reachable and
its `--help` exits 0. If a sub-app import path breaks (say someone renames
`civic_slm.doctor.main`), this is the cheapest test that catches it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from civic_slm.cli import app

runner = CliRunner()


def test_root_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("crawl", "doctor", "eval", "train", "version"):
        assert cmd in r.output


def test_version_prints() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert r.output.strip()  # some version string


@pytest.mark.parametrize(
    "args",
    [
        ["crawl", "--help"],
        ["doctor", "--help"],
        ["eval", "--help"],
        ["eval", "run", "--help"],
        ["eval", "side-by-side", "--help"],
        ["train", "--help"],
        ["train", "cpt", "--help"],
        ["train", "sft", "--help"],
        ["train", "dpo", "--help"],
    ],
)
def test_subcommand_help(args: list[str]) -> None:
    r = runner.invoke(app, args)
    assert r.exit_code == 0, f"{args} failed: {r.output}"
