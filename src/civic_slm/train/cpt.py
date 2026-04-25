"""CPT entry: continued pretraining via mlx_lm.lora.

Why we shell out to `mlx_lm.lora` instead of importing it: MLX-LM's training loop
takes its config as CLI flags, and re-implementing it in-process is more
fragile than just delegating. The downside is we lose direct W&B per-step
hooks; we mitigate by tailing the training output and posting summaries at the
end.

Resume + smoke-test wiring lives in this module rather than in `common.py`
because each stage has slightly different defaults (CPT smoke = 100 iters,
SFT smoke = 50 steps over a packed batch, DPO smoke = 50 steps over
preference pairs).
"""

from __future__ import annotations

from pathlib import Path

import typer

from civic_slm.logging import configure, get_logger
from civic_slm.train.common import TrainConfig, init_wandb
from civic_slm.train.supervisor import echo_command, run_supervised

log = get_logger(__name__)
app = typer.Typer(help="Continued pretraining (mlx_lm.lora --train).")


def build_command(cfg: TrainConfig) -> list[str]:
    iters = cfg.train.iters
    assert iters is not None  # enforced by TrainConfig._check_stage_invariants
    return [
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
        "--num-layers",
        "-1",
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


def _has_existing_adapter(output_dir: Path) -> bool:
    """A mlx_lm adapter dir is considered non-empty if it has any safetensors."""
    if not output_dir.exists():
        return False
    return any(output_dir.glob("*.safetensors"))


@app.command()
def main(
    config: Path = typer.Option(Path("configs/cpt.yaml"), help="Config YAML."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command, don't execute."),
    max_iters_override: int | None = typer.Option(
        None, "--max-iters", help="Override iters (for 100-step smoke run)."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Continue training from an existing adapter at the config's output_dir. "
            "Without this flag, finding an existing adapter aborts to prevent "
            "accidentally overwriting a previous run."
        ),
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help=(
            "Run a 100-iter smoke test (overrides --max-iters). Skips the resume "
            "guard since smoke runs are throwaway. Per CLAUDE.md, do this before "
            "every real CPT run."
        ),
    ),
) -> None:
    configure()
    cfg = TrainConfig.load(config)
    name = init_wandb("cpt", cfg)
    cmd = build_command(cfg)

    if smoke_test:
        max_iters_override = 100
    if max_iters_override is not None:
        idx = cmd.index("--iters")
        cmd[idx + 1] = str(max_iters_override)

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

    log.info("cpt_start", run=name, smoke_test=smoke_test, resume=resume)
    run_supervised(cmd)
    log.info("cpt_done", run=name, output_dir=str(cfg.output_dir))


if __name__ == "__main__":
    app()
