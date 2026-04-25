"""SFT entry: instruction tuning via mlx_lm.lora on chat-formatted pairs."""

from __future__ import annotations

from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, has_existing_adapter, init_wandb
from civic_slm.train.dataset import compute_iters
from civic_slm.train.supervisor import echo_command, run_supervised

log = get_logger(__name__)
app = typer.Typer(help="Instruction tuning (mlx_lm.lora --train, chat format).")


def build_command(cfg: TrainConfig, max_iters: int | None = None) -> list[str]:
    # Schema guarantees these fields exist for an SFT config.
    epochs = cfg.train.epochs
    assert epochs is not None  # enforced by TrainConfig._check_stage_invariants
    if max_iters is not None:
        iters = max_iters
    else:
        iters = compute_iters(
            train_path=cfg.data.train_path,
            epochs=epochs,
            batch_size=cfg.train.batch_size,
            fallback=1000,
        )
    cmd = [
        "mlx_lm.lora",
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
        "--adapter-path",
        str(cfg.output_dir),
        "--lora-rank",
        str(cfg.lora.rank),
        "--steps-per-report",
        str(cfg.logging.steps_per_report),
        "--steps-per-eval",
        str(cfg.logging.steps_per_eval),
        "--save-every",
        str(cfg.logging.steps_per_save),
    ]
    if cfg.train.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    return cmd


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
