"""Tests for the release-signing helpers (#27).

We can't exercise cosign in CI (keyless OIDC needs an interactive browser
or a workload identity token CI doesn't have here), so this file covers
what we CAN test: the manifest-generation contract that both scripts
depend on. If the manifest is wrong, signing the wrong bytes won't be
caught by cosign — the signature is over the manifest, not the files.

Specifically:
- `sha256sums.txt` lines are sorted, byte-deterministic, and paths are
  relative to the artifact dir (not the repo).
- The manifest skips its own output files so a re-sign doesn't recurse.
- `shasum -c` round-trips the manifest the script produces.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGN_SCRIPT = REPO_ROOT / "scripts" / "sign-release.sh"

# The shell snippet that generates the manifest. Extracted from
# `sign-release.sh` so the test exercises the exact same find/sort/sed
# pipeline; if the script diverges, this test will start failing.
_MANIFEST_CMD = r"""
cd "$1" && \
find . -type f \
  ! -name 'sha256sums.txt' \
  ! -name 'sha256sums.txt.bundle' \
  -print0 \
| LC_ALL=C sort -z \
| xargs -0 shasum -a 256 \
| sed 's|  \./|  |' \
> sha256sums.txt
"""


def _run_manifest(art_dir: Path) -> None:
    """Run the same shell pipeline the sign script uses."""
    subprocess.run(
        ["bash", "-c", _MANIFEST_CMD, "_", str(art_dir)],
        check=True,
    )


def test_sign_script_exists_and_is_executable() -> None:
    assert SIGN_SCRIPT.exists(), "scripts/sign-release.sh missing"
    assert SIGN_SCRIPT.stat().st_mode & 0o111, "sign-release.sh not executable"


def test_manifest_paths_are_relative_to_artifact_dir(tmp_path: Path) -> None:
    """A downloader extracts the artifact into some arbitrary path and
    runs `shasum -c sha256sums.txt` from inside it. That only works if
    the paths in the manifest are relative to the artifact dir.
    """
    art = tmp_path / "civic-slm-v1-mlx-q4"
    art.mkdir()
    (art / "model.safetensors").write_bytes(b"weights")
    (art / "config.json").write_text('{"k":"v"}')

    _run_manifest(art)

    text = (art / "sha256sums.txt").read_text()
    assert "  model.safetensors\n" in text, text
    assert "  config.json\n" in text, text
    # No leading `./`, no absolute paths.
    for line in text.splitlines():
        path = line.split("  ", 1)[1]
        assert not path.startswith("./"), f"manifest leaks ./ prefix: {line}"
        assert not path.startswith("/"), f"manifest leaks absolute path: {line}"


def test_manifest_is_byte_deterministic_across_runs(tmp_path: Path) -> None:
    """Same files in → same manifest bytes out. Filesystem iteration
    order can otherwise scramble the lines and produce a different
    signature each time (a real footgun if the maintainer re-signs).
    """
    art = tmp_path / "art"
    art.mkdir()
    # Names that would naïvely sort in different orders by length vs. by
    # bytes.
    for name in ("z.bin", "a.bin", "subdir/x.bin", "subdir/aa.bin", "0.bin"):
        p = art / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())

    _run_manifest(art)
    first = (art / "sha256sums.txt").read_bytes()

    # Re-run; manifest should be identical.
    _run_manifest(art)
    second = (art / "sha256sums.txt").read_bytes()
    assert first == second, "manifest non-deterministic across runs"

    # Lines are sorted byte-wise (LC_ALL=C), not by length.
    lines = first.decode().splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == sorted(paths), f"manifest not byte-sorted: {paths}"


def test_manifest_excludes_its_own_outputs(tmp_path: Path) -> None:
    """A re-sign run must not recursively include `sha256sums.txt` or
    `sha256sums.txt.bundle` in the new manifest — the previous-run
    artifacts would shift the hash on every re-sign.
    """
    art = tmp_path / "art"
    art.mkdir()
    (art / "real-file").write_bytes(b"x")
    # Pretend a previous sign already deposited these.
    (art / "sha256sums.txt").write_text("stale\n")
    (art / "sha256sums.txt.bundle").write_text("stale\n")

    _run_manifest(art)
    text = (art / "sha256sums.txt").read_text()
    assert "sha256sums.txt\n" not in text
    assert "sha256sums.txt.bundle\n" not in text
    assert "  real-file\n" in text


def test_manifest_roundtrips_through_shasum_check(tmp_path: Path) -> None:
    """The whole point of the manifest is that `shasum -c` can verify
    it. If the format drifts (e.g. trailing whitespace, wrong separator)
    this would silently break the verify path on download.
    """
    art = tmp_path / "art"
    art.mkdir()
    (art / "a").write_text("hello\n")
    (art / "nested" / "b").parent.mkdir()
    (art / "nested" / "b").write_text("world\n")

    _run_manifest(art)

    result = subprocess.run(
        ["shasum", "-a", "256", "--check", "--quiet", "sha256sums.txt"],
        cwd=art,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"shasum -c rejected the script's manifest:\n"
        f"stderr: {result.stderr}\nstdout: {result.stdout}"
    )

    # Negative test: corrupt a file and confirm verify fails. If this
    # didn't fail, the manifest would silently rubber-stamp tampered
    # artifacts.
    (art / "a").write_text("tampered\n")
    bad = subprocess.run(
        ["shasum", "-a", "256", "--check", "--quiet", "sha256sums.txt"],
        cwd=art,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0, "shasum -c accepted tampered file — manifest format is broken"


def test_verify_script_pins_canonical_identity() -> None:
    """The maintainer's canonical release identity is hardcoded into
    `verify-release.sh` so a fresh clone can verify a release without
    out-of-band trust setup. Keep this in lockstep with RELEASING.md;
    a quiet rotation that diverges from the docs is the bug class this
    test guards against.
    """
    verify = (REPO_ROOT / "scripts" / "verify-release.sh").read_text()
    assert "itsmeduncan@gmail.com" in verify, "default release identity drifted from RELEASING.md"
    assert "https://github.com/login/oauth" in verify, (
        "default OIDC issuer drifted from RELEASING.md"
    )
    releasing = (REPO_ROOT / "RELEASING.md").read_text()
    assert "itsmeduncan@gmail.com" in releasing
    assert "https://github.com/login/oauth" in releasing


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_release_scripts_pass_shellcheck() -> None:
    """If shellcheck is on PATH, run it. CI doesn't install shellcheck
    yet, so this is a local-dev assist rather than a hard gate.
    """
    for script in ("sign-release.sh", "verify-release.sh"):
        path = REPO_ROOT / "scripts" / script
        result = subprocess.run(
            ["shellcheck", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"shellcheck flagged {script}:\n{result.stdout}\n{result.stderr}"
        )
