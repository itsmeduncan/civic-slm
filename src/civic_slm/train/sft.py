"""SFT entry: instruction tuning via mlx_lm.lora on chat-formatted pairs."""

from __future__ import annotations

from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import (
    TrainConfig,
    has_existing_adapter,
    init_wandb,
    write_mlx_lora_config,
)
from civic_slm.train.dataset import compute_iters
from civic_slm.train.supervisor import echo_command, run_supervised

log = get_logger(__name__)
app = typer.Typer(help="Instruction tuning (mlx_lm.lora --train, chat format).")


def build_command(cfg: TrainConfig, max_iters: int | None = None) -> list[str]:
    """Same shape as cpt.build_command — materialize a YAML and call `-c`."""
    epochs = cfg.train.epochs
    assert epochs is not None  # enforced by TrainConfig._check_stage_invariants
    iters = max_iters or compute_iters(
        train_path=cfg.data.train_path,
        epochs=epochs,
        batch_size=cfg.train.batch_size,
        fallback=1000,
    )
    yaml_path = write_mlx_lora_config(
        cfg,
        iters=iters,
        out_path=cfg.output_dir / "mlx_lm_config.yaml",
    )
    return ["mlx_lm.lora", "-c", str(yaml_path)]


@app.command()
def main(
    config: Path = typer.Option(Path("configs/sft.yaml")),
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
        help="Run a 50-step SFT smoke test. Skips the resume guard.",
    ),
) -> None:
    configure()
    cfg = TrainConfig.load(config)
    name = init_wandb("sft", cfg)

    if smoke_test:
        max_iters = 50
    cmd = build_command(cfg, max_iters=max_iters)

    if dry_run:
        echo_command(cmd)
        return

    if not smoke_test and has_existing_adapter(cfg.output_dir) and not resume:
        typer.echo(
            f"refusing to overwrite existing adapter at {cfg.output_dir}. "
            "Re-run with --resume to continue training, or move/delete the "
            "directory to start fresh.",
            err=True,
        )
        raise typer.Exit(code=2)

    log.info("sft_start", run=name, smoke_test=smoke_test, resume=resume)
    run_supervised(cmd)
    log.info("sft_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
