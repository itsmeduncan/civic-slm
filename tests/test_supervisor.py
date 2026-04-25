"""Tests for the trainer-subprocess supervisor.

We deliberately don't shell out to mlx_lm — these tests use short shell
commands that are guaranteed to be on every system, plus a mocked
`subprocess.Popen` for the signal-forwarding path.

Cross-process signal propagation is verified by a manual smoke test
documented in `RELEASING.md` rather than in CI; coordinating SIGINTs
across pytest workers is too flaky to be load-bearing.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

from civic_slm.train.supervisor import TrainerError, run_supervised


def test_clean_exit_zero_returns_zero() -> None:
    assert run_supervised(["true"]) == 0


def test_nonzero_exit_raises_trainer_error() -> None:
    with pytest.raises(TrainerError):
        run_supervised(["false"])


def test_signal_forwarding_calls_send_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake Popen and post a SIGTERM via the supervisor's handler."""
    sent: list[int] = []

    class _FakePopen:
        pid = 99999
        returncode: int | None = None

        def __init__(self, _cmd: object) -> None:
            self._calls = 0

        def poll(self) -> int | None:
            # First two polls return None (running), then the test posts
            # SIGTERM via the registered handler, then we return 0.
            self._calls += 1
            if self._calls >= 4:
                self.returncode = 0
                return 0
            return None

        def send_signal(self, signum: int) -> None:
            sent.append(signum)

        def kill(self) -> None:  # pragma: no cover — not exercised here
            self.returncode = -9

        def wait(self) -> None:  # pragma: no cover
            return None

    monkeypatch.setattr("civic_slm.train.supervisor.subprocess.Popen", _FakePopen)

    # Schedule a SIGTERM to ourselves so the supervisor's handler runs and
    # forwards it to our fake child.
    monkeypatch.setattr("civic_slm.train.supervisor.time.sleep", _make_sleep_that_signals())

    code = run_supervised(["true"])
    assert code == 0
    assert signal.SIGTERM in sent


def _make_sleep_that_signals() -> Any:
    """Return a fake `time.sleep` that, on its second call, raises SIGTERM
    on the current process so the supervisor's signal handler runs."""
    state = {"calls": 0}

    def _sleep(_seconds: float) -> None:
        state["calls"] += 1
        if state["calls"] == 2:
            import os

            os.kill(os.getpid(), signal.SIGTERM)

    return _sleep


def test_resume_guard_blocks_overwrite_without_flag(tmp_path: Path) -> None:
    """The CPT entry refuses to overwrite an existing adapter without --resume."""
    from civic_slm.train.cpt import _has_existing_adapter

    output_dir = tmp_path / "qwen-civic-cpt"
    assert _has_existing_adapter(output_dir) is False
    output_dir.mkdir()
    assert _has_existing_adapter(output_dir) is False
    (output_dir / "adapters.safetensors").write_bytes(b"")
    assert _has_existing_adapter(output_dir) is True
