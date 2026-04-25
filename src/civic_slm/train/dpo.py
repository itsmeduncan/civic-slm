"""DPO entry: preference optimization via mlx_lm.dpo.

Note: mlx_lm.dpo support is newer than the LoRA path. If the binary isn't
available, surface that with an actionable message rather than failing
mid-train.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, init_wandb
from civic_slm.train.cpt import _has_existing_adapter
from civic_slm.train.dataset import compute_iters
from civic_slm.train.supervisor import echo_command, run_supervised

log = get_logger(__name__)
app = typer.Typer(help="DPO training (mlx_lm.dpo).")


def build_command(cfg: TrainConfig, max_iters: int | None = None) -> list[str]:
    epochs = cfg.train.epochs or 1
    beta = cfg.train.beta
    assert beta is not None  # enforced by TrainConfig._check_stage_invariants
    if max_iters is not None:
        iters = max_iters
    else:
        iters = compute_iters(
            train_path=cfg.data.train_path,
            epochs=epochs,
            batch_size=cfg.train.batch_size,
            fallback=500,
        )
    return [
        "mlx_lm.dpo",
        "--model",
        cfg.base_model,
        "--train",
        "--data",
        str(cfg.data.train_path.parent),
        "--iters",
        str(iters),
        "--batch-size",
        str(cfg.train.batch_size),
        "--max-seq-length",
        str(cfg.train.max_seq_length),
        "--learning-rate",
        str(cfg.train.learning_rate),
        "--beta",
        str(beta),
        "--adapter-path",
        str(cfg.output_dir),
    ]


@app.command()
def main(
    config: Path = typer.Option(Path("configs/dpo.yaml")),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_iters: int | None = typer.Option(None, "--max-iters"),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue training from the existing adapter at output_dir.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Run a 50-step DPO smoke test. Skips the resume guard.",
    ),
) -> None:
    configure()
    if shutil.which("mlx_lm.dpo") is None and not dry_run:
        typer.echo(
            "mlx_lm.dpo not found on PATH. Install MLX-LM with DPO support, "
            "or skip DPO for v0 (CPT+SFT is acceptable per CLAUDE.md).",
            err=True,
        )
        raise typer.Exit(code=2)
    cfg = TrainConfig.load(config)
    name = init_wandb("dpo", cfg)

    if smoke_test:
        max_iters = 50
    cmd = build_command(cfg, max_iters=max_iters)

    if dry_run:
        echo_command(cmd)
        return

    if not smoke_test and _has_existing_adapter(cfg.output_dir) and not resume:
        typer.echo(
            f"refusing to overwrite existing adapter at {cfg.output_dir}. "
            "Re-run with --resume to continue training, or move/delete the "
            "directory to start fresh.",
            err=True,
        )
        raise typer.Exit(code=2)

    log.info("dpo_start", run=name, smoke_test=smoke_test, resume=resume)
    run_supervised(cmd)
    log.info("dpo_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
