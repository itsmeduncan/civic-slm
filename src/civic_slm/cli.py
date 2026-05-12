"""Top-level Typer app. All stage CLIs hang off this entry point.

Stage CLIs each define a `main()` function with the actual options; we register
those functions directly here so `civic-slm eval run --model ...` works without
the awkward `civic-slm eval run main --model ...` Typer creates by default for
multi-command sub-apps.
"""

from __future__ import annotations

import typer

from civic_slm import __version__
from civic_slm.doctor import main as doctor_main
from civic_slm.eval.runner import main as eval_run_main
from civic_slm.eval.seed import main as eval_seed_main
from civic_slm.eval.side_by_side import main as eval_sxs_main
from civic_slm.ingest.crawl import main as crawl_main
from civic_slm.ingest.crawl import videos_main as crawl_videos_main
from civic_slm.ingest.process import main as process_main
from civic_slm.synth.cli import main as synth_main
from civic_slm.synth.review import main as review_main
from civic_slm.train.cpt import main as cpt_main
from civic_slm.train.dpo import main as dpo_main
from civic_slm.train.merge import main as merge_main
from civic_slm.train.prepare_cpt import main as prepare_cpt_main
from civic_slm.train.prepare_sft import main as prepare_sft_main
from civic_slm.train.sft import main as sft_main

app = typer.Typer(
    name="civic-slm",
    help="Civic SLM — Qwen fine-tune for U.S. local-government documents.",
    no_args_is_help=True,
)

eval_app = typer.Typer(help="Evaluation runners.", no_args_is_help=True)
eval_app.command("run")(eval_run_main)
eval_app.command("side-by-side")(eval_sxs_main)
eval_app.command("seed")(eval_seed_main)

train_app = typer.Typer(help="Training stages.", no_args_is_help=True)
train_app.command("cpt")(cpt_main)
train_app.command("sft")(sft_main)
train_app.command("dpo")(dpo_main)

app.command("crawl")(crawl_main)
app.command("crawl-videos")(crawl_videos_main)
app.command("process")(process_main)
app.command("synth")(synth_main)
app.command("review-sft")(review_main)
app.command("prepare-cpt")(prepare_cpt_main)
app.command("prepare-sft")(prepare_sft_main)
app.command("merge")(merge_main)
app.command("doctor")(doctor_main)
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")


@app.callback()
def _root() -> None:  # pyright: ignore[reportUnusedFunction]
    """Root callback so subcommands aren't collapsed into the root."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
