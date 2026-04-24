"""CPT entry: continued pretraining via mlx_lm.lora.

Why we shell out to `mlx_lm.lora` instead of importing it: MLX-LM's training loop
takes its config as CLI flags, and re-implementing it in-process is more
fragile than just delegating. The downside is we lose direct W&B per-step
hooks; we mitigate by tailing the training output and posting summaries at the
end.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, init_wandb

log = get_logger(__name__)
app = typer.Typer(help="Continued pretraining (mlx_lm.lora --train).")


def build_command(cfg: TrainConfig) -> list[str]:
    raw = cfg.raw
    return [
        "mlx_lm.lora",
        "--model",
        cfg.base_model,
        "--train",
        "--data",
        str(Path(raw["data"]["train_path"]).parent),
        "--iters",
        str(raw["train"]["iters"]),
        "--batch-size",
        str(raw["train"]["batch_size"]),
        "--max-seq-length",
        str(raw["train"]["max_seq_length"]),
        "--learning-rate",
        str(raw["train"]["learning_rate"]),
        "--num-layers",
        "-1",
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


@app.command()
def main(
    config: Path = typer.Option(Path("configs/cpt.yaml"), help="Config YAML."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command, don't execute."),
    max_iters_override: int | None = typer.Option(
        None, "--max-iters", help="Override iters (for 100-step smoke run)."
    ),
) -> None:
    configure()
    cfg = TrainConfig.load(config)
    name = init_wandb("cpt", cfg)
    cmd = build_command(cfg)
    if max_iters_override is not None:
        idx = cmd.index("--iters")
        cmd[idx + 1] = str(max_iters_override)
    log.info("cpt_start", run=name, cmd=shlex.join(cmd))
    if dry_run:
        typer.echo(shlex.join(cmd))
        return
    subprocess.run(cmd, check=True)
    log.info("cpt_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
