"""DPO entry: preference optimization via mlx_lm.dpo.

Note: mlx_lm.dpo support is newer than the LoRA path. If the binary isn't
available, surface that with an actionable message rather than failing
mid-train.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, init_wandb
from civic_slm.train.dataset import compute_iters

log = get_logger(__name__)
app = typer.Typer(help="DPO training (mlx_lm.dpo).")


def build_command(cfg: TrainConfig, max_iters: int | None = None) -> list[str]:
    raw = cfg.raw
    if max_iters is not None:
        iters = max_iters
    else:
        iters = compute_iters(
            train_path=Path(raw["data"]["train_path"]),
            epochs=int(raw["train"].get("epochs", 1)),
            batch_size=int(raw["train"].get("batch_size", 1)),
            fallback=500,
        )
    return [
        "mlx_lm.dpo",
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
        "--beta",
        str(raw["train"]["beta"]),
        "--adapter-path",
        str(cfg.output_dir),
    ]


@app.command()
def main(
    config: Path = typer.Option(Path("configs/dpo.yaml")),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_iters: int | None = typer.Option(None, "--max-iters"),
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
    cmd = build_command(cfg, max_iters=max_iters)
    log.info("dpo_start", run=name, cmd=shlex.join(cmd))
    if dry_run:
        typer.echo(shlex.join(cmd))
        return
    subprocess.run(cmd, check=True)
    log.info("dpo_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
