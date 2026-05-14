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
    # Number of transformer layers to apply LoRA to. -1 = every layer (the
    # original sweet-spot for small bases). On the 27B Qwen3.5 hybrid base
    # this overflows Metal at r≥32 / seq≥1024; cap to 8 unless your base is
    # smaller. Mirrors mlx_lm.lora's `--num-layers` (default 16).
    num_layers: int = 16


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


def write_mlx_lora_config(
    cfg: TrainConfig,
    *,
    iters: int,
    out_path: Path,
) -> Path:
    """Write an mlx_lm.lora-compatible YAML config from our TrainConfig.

    mlx_lm 0.31+ took LoRA hyperparameters off the CLI surface — `--lora-rank`,
    `--target-modules`, `--dropout` are gone; they live in a YAML config
    consumed via `mlx_lm.lora -c <path>`. We materialize that YAML at runtime
    so callers keep the civic-slm-shaped TrainConfig as the source of truth.

    The output YAML mirrors `mlx_lm.lora.CONFIG_DEFAULTS` (see
    `mlx_lm/lora.py`). `scale` is `alpha / rank` (mlx_lm scales LoRA Δ by
    this factor at apply time, where HF/PEFT exposes alpha directly).
    Validation data is included only when `valid_path` is set on the config.

    `out_path` is the YAML to write. Returned for chaining; the caller is
    responsible for cleanup if it's a temp file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mlx_lm.lora reads --data as a directory containing train.jsonl /
    # valid.jsonl. The civic-slm TrainConfig points train_path at the file
    # itself; pass the parent.
    data_dir = cfg.data.train_path.parent
    payload: dict[str, Any] = {
        "model": cfg.base_model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": 0,
        "num_layers": cfg.train.num_layers,
        "batch_size": cfg.train.batch_size,
        "iters": iters,
        # -1 → use the full validation set every eval.
        "val_batches": -1,
        "learning_rate": cfg.train.learning_rate,
        "steps_per_report": cfg.logging.steps_per_report,
        "steps_per_eval": cfg.logging.steps_per_eval,
        "adapter_path": str(cfg.output_dir),
        "save_every": cfg.logging.steps_per_save,
        "max_seq_length": cfg.train.max_seq_length,
        "grad_checkpoint": cfg.train.grad_checkpoint,
        "grad_accumulation_steps": 1,
        "lora_parameters": {
            "rank": cfg.lora.rank,
            "dropout": cfg.lora.dropout,
            # alpha is exposed as `scale = alpha / rank` in mlx_lm.
            "scale": cfg.lora.alpha / cfg.lora.rank,
        },
    }
    if cfg.lora.target_modules and cfg.lora.target_modules != "all-linear":
        # mlx_lm calls them `keys`; "all-linear" is the default (no keys
        # restriction — every linear projection gets LoRA).
        payload["lora_parameters"]["keys"] = cfg.lora.target_modules
    if cfg.train.lr_schedule == "cosine" and cfg.train.iters:
        # mlx_lm consumes lr_schedule as {name, arguments, warmup}. cosine_decay
        # arguments are [initial_lr, num_steps].
        warmup = cfg.train.warmup_steps or (int((cfg.train.warmup_ratio or 0.0) * cfg.train.iters))
        payload["lr_schedule"] = {
            "name": "cosine_decay",
            "arguments": [cfg.train.learning_rate, iters],
            "warmup": int(warmup),
        }
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out_path


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
