"""Shared training helpers — config loading, run naming, W&B init.

Why a thin shared layer instead of one big trainer: each stage has a different
data shape (raw text, chat pairs, preference triples) and `mlx_lm` exposes
distinct entrypoints. Sharing config parsing + naming keeps logging and W&B
naming consistent without forcing a one-size-fits-all trainer.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        return cls(raw=data, path=path)

    @property
    def stage(self) -> str:
        return str(self.raw["stage"])

    @property
    def base_model(self) -> str:
        return str(self.raw["base_model"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["output_dir"])


def git_sha(short: int = 8) -> str:
    """Return the current git SHA, or 'unknown' if not in a git tree."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()[:short]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_name(stage: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stage}-{git_sha()}-{ts}"


def init_wandb(stage: str, cfg: TrainConfig) -> str:
    """Init W&B if available; returns the run name. No-op if wandb isn't installed.

    Only `ImportError` is swallowed — wandb is optional. Runtime failures
    (auth, bad project) are logged as warnings so you know why your run
    didn't land in the dashboard, but they don't abort training.
    """
    from civic_slm.logging import get_logger

    log = get_logger(__name__)
    name = run_name(stage)
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        return name
    try:
        wandb.init(  # pyright: ignore[reportUnknownMemberType]
            project="civic-slm",
            name=name,
            config=cfg.raw,
            reinit=True,
        )
    except Exception as exc:  # wandb isn't load-bearing; log + continue
        log.warning("wandb_init_failed", stage=stage, error=str(exc))
    return name
