"""SFT entry: instruction tuning via mlx_lm.lora on chat-formatted pairs."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, init_wandb
from civic_slm.train.dataset import compute_iters

log = get_logger(__name__)
app = typer.Typer(help="Instruction tuning (mlx_lm.lora --train, chat format).")


def build_command(cfg: TrainConfig, max_iters: int | None = None) -> list[str]:
    raw = cfg.raw
    if max_iters is not None:
        iters = max_iters
    else:
        iters = compute_iters(
            train_path=Path(raw["data"]["train_path"]),
            epochs=int(raw["train"]["epochs"]),
            batch_size=int(raw["train"].get("batch_size", 1)),
            fallback=1000,
        )
    cmd = [
        "mlx_lm.lora",
        "--model",
        cfg.base_model,
        "--train",
        "--data",
        str(Path(raw["data"]["train_path"]).parent),
        "--iters",
        str(iters),
        "--batch-size",
        str(raw["train"]["batch_size"]),
        "--max-seq-length",
        str(raw["train"]["max_seq_length"]),
        "--learning-rate",
        str(raw["train"]["learning_rate"]),
        "--adapter-path",
        str(cfg.output_dir),
        "--lora-rank",
        str(raw["lora"]["rank"]),
        "--steps-per-report",
        str(raw["logging"]["steps_per_report"]),
        "--steps-per-eval",
        str(raw["logging"]["steps_per_eval"]),
        "--save-every",
        str(raw["logging"]["steps_per_save"]),
    ]
    if raw["train"].get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")
    return cmd


@app.command()
def main(
    config: Path = typer.Option(Path("configs/sft.yaml")),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_iters: int | None = typer.Option(None, "--max-iters"),
) -> None:
    configure()
    cfg = TrainConfig.load(config)
    name = init_wandb("sft", cfg)
    cmd = build_command(cfg, max_iters=max_iters)
    log.info("sft_start", run=name, cmd=shlex.join(cmd))
    if dry_run:
        typer.echo(shlex.join(cmd))
        return
    subprocess.run(cmd, check=True)
    log.info("sft_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
