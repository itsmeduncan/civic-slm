"""Shared training helpers — config loading, run naming, W&B init.

Why a thin shared layer instead of one big trainer: each stage has a different
data shape (raw text, chat pairs, preference triples) and `mlx_lm` exposes
distinct entrypoints. Sharing config parsing + naming keeps logging and W&B
naming consistent without forcing a one-size-fits-all trainer.

Why Pydantic on the YAML: a missing key in `configs/sft.yaml` previously surfaced
as a `KeyError` deep inside `build_command`. Validating at load time with a
frozen model makes typos fail fast with a useful message and keeps the rest of
the codebase consistent with `ARCHITECTURE.md` ("if it doesn't validate, it
doesn't land").
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _DataSection(_Frozen):
    train_path: Path
    valid_path: Path | None = None
    format: Literal["text", "chat", "dpo"]


class _LoraSection(_Frozen):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    target_modules: str = "all-linear"


class _TrainSection(_Frozen):
    """Training hyperparameters.

    `iters` is required for CPT (which trains on a raw text corpus where
    epoch boundaries are arbitrary); `epochs` is required for SFT/DPO (which
    have well-defined dataset cardinality). The model-level validator below
    enforces this so missing keys fail loudly, not silently with a default.
    """

    iters: int | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_checkpoint: bool = False
    max_seq_length: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    warmup_steps: int | None = Field(default=None, ge=0)
    packing: bool = False
    beta: float | None = Field(default=None, gt=0.0)


class _LoggingSection(_Frozen):
    project: str = "civic-slm"
    steps_per_eval: int = Field(default=100, gt=0)
    steps_per_report: int = Field(default=10, gt=0)
    steps_per_save: int = Field(default=500, gt=0)


class TrainConfig(_Frozen):
    """Frozen, validated training config — one schema for CPT, SFT, and DPO.

    Loaded from `configs/{cpt,sft,dpo}.yaml` via `TrainConfig.load(path)`.
    `path` is captured for logging only and is not part of the schema.
    """

    stage: Literal["cpt", "sft", "dpo"]
    base_model: str = Field(min_length=1)
    # Optional HF Hub revision pin. Either a branch (`main`), a tag, or a
    # 40-char commit SHA. Pinning a SHA is the integrity story — see
    # MODEL_CARD.md "Model details" for the recommended posture.
    base_model_revision: str | None = None
    data: _DataSection
    lora: _LoraSection
    train: _TrainSection
    logging: _LoggingSection = _LoggingSection()
    output_dir: Path

    # Captured at load time for diagnostics; not validated, not in YAML.
    path: Path | None = None

    @model_validator(mode="after")
    def _check_stage_invariants(self) -> TrainConfig:
        if self.stage == "cpt" and self.train.iters is None:
            raise ValueError("CPT config must set `train.iters` (raw-text training has no epochs).")
        if self.stage in {"sft", "dpo"} and self.train.epochs is None:
            raise ValueError(f"{self.stage.upper()} config must set `train.epochs`.")
        if self.stage == "dpo" and self.train.beta is None:
            raise ValueError("DPO config must set `train.beta` (preference temperature).")
        return self

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Load and validate a YAML training config. Raises `ConfigError` on bad input."""
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        try:
            cfg = cls.model_validate({**data, "path": path})
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid training config at {path}:\n{exc}\n\n"
                "Check the field names and types against `configs/cpt.yaml`, "
                "`configs/sft.yaml`, and `configs/dpo.yaml` — those are the "
                "canonical examples."
            ) from exc
        return cfg

    @property
    def raw(self) -> dict[str, Any]:
        """Backwards-compatible dict view used by `init_wandb` and trainer wrappers."""
        return self.model_dump(mode="json", exclude={"path"})


class ConfigError(ValueError):
    """Raised when a training-config YAML fails schema validation."""


def has_existing_adapter(output_dir: Path) -> bool:
    """True when `output_dir` already contains an mlx_lm LoRA adapter file.

    Used by every trainer wrapper to decide whether `--resume` is required
    before launching. Lives in `common` (not `cpt.py`) because all three
    stage modules need it and a cross-module private import is awkward.
    """
    if not output_dir.exists():
        return False
    return any(output_dir.glob("*.safetensors"))


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
            project=cfg.logging.project,
            name=name,
            config=cfg.raw,
            reinit=True,
        )
    except Exception as exc:  # wandb isn't load-bearing; log + continue
        log.warning("wandb_init_failed", stage=stage, error=str(exc))
    return name
