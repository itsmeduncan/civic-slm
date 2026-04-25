"""Subprocess supervisor for the mlx_lm trainer wrappers.

`subprocess.run(..., check=True)` is the wrong primitive for a multi-hour
training job: a Ctrl-C lands on the parent and the child is left to die at
its leisure (sometimes leaving a corrupt adapter checkpoint), and a SIGTERM
from the OS just kills the parent without giving mlx_lm a chance to flush.

This module provides a thin supervisor that:

  - launches the trainer as a foreground child via `subprocess.Popen`;
  - propagates SIGTERM and SIGINT to the child so a Ctrl-C cleanly stops
    training (mlx_lm flushes its checkpoint on SIGTERM);
  - waits up to `kill_grace_seconds` after sending SIGTERM before escalating
    to SIGKILL;
  - returns the child's exit code (or raises `TrainerError` with the original
    signal as context).

The supervisor is intentionally not async — training is a long blocking
operation, and the simpler signal-handler-based approach is easier to
reason about than juggling asyncio task cancellation across signal events.
"""

from __future__ import annotations

import contextlib
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

from civic_slm.logging import get_logger

log = get_logger(__name__)


class TrainerError(RuntimeError):
    """Raised when the trainer subprocess exits non-zero or is signalled."""


def run_supervised(
    cmd: Sequence[str],
    *,
    kill_grace_seconds: float = 10.0,
) -> int:
    """Run `cmd` and propagate SIGTERM/SIGINT to it. Return the exit code.

    Raises `TrainerError` if the child exits non-zero. A clean Ctrl-C
    (SIGINT propagated and child exits with the conventional 130 code,
    or with 0 after flushing) does **not** raise — it's a normal
    early-stop and the caller decides how to handle it.

    The loop polls every 250ms; each poll also gives Python a chance to
    process pending signals. Lower poll intervals burn CPU; higher ones
    delay propagation.
    """
    log.info("trainer_start", cmd=cmd)
    proc = subprocess.Popen(cmd)
    forwarded: list[int] = []

    def _forward(signum: int, _frame: object) -> None:
        forwarded.append(signum)
        log.warning("trainer_signal_forwarded", signum=signum, pid=proc.pid)
        # Child may already be gone — that's fine.
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(signum)

    prev_term = signal.signal(signal.SIGTERM, _forward)
    prev_int = signal.signal(signal.SIGINT, _forward)
    try:
        while proc.poll() is None:
            time.sleep(0.25)
            if forwarded and proc.poll() is None:
                # We've sent at least one signal; if the child hasn't exited
                # within the grace window, escalate.
                deadline = time.monotonic() + kill_grace_seconds
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.25)
                if proc.poll() is None:
                    log.error("trainer_kill_after_grace", grace_s=kill_grace_seconds)
                    proc.kill()
                    proc.wait()
                break
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)

    code = proc.returncode
    if forwarded:
        log.info("trainer_signalled_exit", code=code, signum=forwarded[0])
        # 0 (graceful shutdown) and 130 (SIGINT-conventional) are normal
        # for an operator-initiated stop; anything else is an error.
        if code not in (0, 130, -signal.SIGTERM, -signal.SIGINT):
            raise TrainerError(f"trainer exited {code} after signal {forwarded[0]}")
        return code
    if code != 0:
        raise TrainerError(f"trainer exited non-zero ({code}); see log above")
    return code


def echo_command(cmd: Sequence[str]) -> None:
    """Print a shell-quoted version of `cmd` for `--dry-run`."""
    import shlex

    sys.stdout.write(shlex.join(cmd) + "\n")
