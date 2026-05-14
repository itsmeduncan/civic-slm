"""`civic-slm train-jurisdiction <slug>` — one command, end-to-end pipeline.

Phase 2 of the snazzy-tinkering-stardust plan. The pieces (crawl, process,
synth, prepare-cpt, train cpt, fuse, prepare-sft, train sft, fuse, quantize,
eval) all already exist as standalone commands — this module is a thin
composer that runs them in order with sensible defaults for a 128GB Mac
and a per-stage `status.json` so a Ctrl-C plus `--resume` picks back up
without re-doing finished work.

Why a composer and not just docs: a fresh maintainer following USAGE.md
has to type 11 commands, remember which output paths to thread between
them, and not skip the contamination guard. Each step is a place to fumble.
`train-jurisdiction santa-monica` is the path that actually ships.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml

from civic_slm.config import settings
from civic_slm.logging import configure, get_logger

log = get_logger(__name__)


_STAGES = (
    "crawl",
    "process",
    "synth",
    "prepare_cpt",
    "prepare_sft",
    "train_cpt",
    "fuse_cpt",
    "train_sft",
    "fuse_v1",
    "quantize",
    "eval",
)


class StageError(RuntimeError):
    """A pipeline stage failed and the user wants to know which one."""


def _status_path(slug: str, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"{slug}-pipeline" / "status.json"


def _load_status(slug: str, artifacts_dir: Path) -> dict[str, str]:
    path = _status_path(slug, artifacts_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_status(slug: str, artifacts_dir: Path, status: dict[str, str]) -> None:
    path = _status_path(slug, artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _stage_done(stage: str, status: dict[str, str]) -> bool:
    return status.get(stage, "").startswith("done")


def _mark(stage: str, status: dict[str, str], slug: str, artifacts_dir: Path) -> None:
    status[stage] = f"done@{datetime.now(UTC).isoformat()}"
    _save_status(slug, artifacts_dir, status)


def _run_cli(args: list[str]) -> None:
    """Run `uv run civic-slm <args>` as a subprocess and propagate exit code.

    Subprocessing is the right call here even though we're inside the same
    Python process: each stage's Typer command has its own logging/config
    setup, and a clean process boundary keeps an MLX OOM from corrupting
    the supervisor's state.
    """
    cmd = [sys.executable, "-m", "civic_slm", *args]
    log.info("pipeline_run", args=args)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise StageError(f"`civic-slm {' '.join(args)}` exited {proc.returncode}")


def _write_stage_config(
    src_path: Path,
    out_path: Path,
    *,
    base_model: str,
    train_path: Path,
    valid_path: Path,
    output_dir: Path,
) -> Path:
    """Materialize a stage config from a template by overriding the per-slug fields."""
    with src_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["base_model"] = base_model
    cfg["data"]["train_path"] = str(train_path)
    cfg["data"]["valid_path"] = str(valid_path)
    cfg["output_dir"] = str(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out_path


def main(
    slug: str = typer.Argument(..., help="Jurisdiction slug (must exist as a recipe)."),
    base_model: str = typer.Option(
        "/Users/duncan/.lmstudio/models/Brooooooklyn/Qwen3.6-27B-UD-Q6_K_XL-mlx",
        "--base-model",
        help=(
            "Local path or HF repo of the base model. Must match what mlx_lm.lora "
            "accepts. Default points at the maintainer's LM Studio cache; "
            "override on machines without that cache."
        ),
    ),
    max_docs: int = typer.Option(
        50, "--max-docs", help="Max docs to crawl (capped by the site's archive)."
    ),
    since: str = typer.Option(
        "2024-01-01",
        "--since",
        help="ISO date for the earliest meeting to crawl.",
    ),
    cpt_iters: int = typer.Option(
        200, "--cpt-iters", help="CPT iterations. 200 is a good fit for a ~30-chunk corpus."
    ),
    n_per_chunk: int = typer.Option(3, "--n-per-chunk", help="Synth examples per (chunk, task)."),
    synth_rounds: int = typer.Option(
        1,
        "--synth-rounds",
        min=1,
        help=(
            "How many synth passes to run. Each round stacks on the previous "
            "(see `civic-slm synth --rounds`). For a ~5k-example corpus from "
            "~35 chunks, try `--synth-rounds 12 --n-per-chunk 3`."
        ),
    ),
    skip_quantize: bool = typer.Option(
        False, "--skip-quantize", help="Stop after the v1 fuse; don't emit MLX-q4."
    ),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Stop before the final eval pass."),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Pick up from the last completed stage (per status.json).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the planned stages and exit."),
) -> None:
    """Run the full per-jurisdiction training pipeline.

    Stages, in order: crawl → process → synth → prepare-cpt → prepare-sft →
    train cpt → fuse cpt → train sft → fuse v1 → quantize → eval.

    Each stage records to `artifacts/<slug>-pipeline/status.json`; a kill +
    rerun skips finished stages unless `--no-resume` is set.
    """
    configure()
    artifacts_dir = settings().artifacts_dir
    data_dir = settings().data_dir
    status = _load_status(slug, artifacts_dir) if resume else {}

    cpt_adapter = artifacts_dir / f"{slug}-cpt"
    cpt_fused = artifacts_dir / f"{slug}-cpt-fused"
    sft_adapter = artifacts_dir / f"{slug}-sft"
    v1_fused = artifacts_dir / f"{slug}-v1-fused"
    v1_quant = artifacts_dir / f"{slug}-v1-mlx-q4"

    cpt_config_template = Path("configs/jurisdiction-default.cpt.yaml")
    sft_config_template = Path("configs/jurisdiction-default.sft.yaml")
    cpt_config_out = artifacts_dir / f"{slug}-pipeline" / "cpt.yaml"
    sft_config_out = artifacts_dir / f"{slug}-pipeline" / "sft.yaml"

    cpt_data_dir = data_dir / "processed" / f"{slug}-cpt"
    sft_data_dir = data_dir / "sft" / f"{slug}-sft"

    plan: list[tuple[str, list[str]]] = [
        ("crawl", ["crawl", slug, "--max", str(max_docs), "--since", since]),
        ("process", ["process", slug]),
        (
            "synth",
            [
                "synth",
                slug,
                "--n-per-chunk",
                str(n_per_chunk),
                "--rounds",
                str(synth_rounds),
            ],
        ),
        ("prepare_cpt", ["prepare-cpt", slug, "--out-dir", str(cpt_data_dir)]),
        (
            "prepare_sft",
            [
                "prepare-sft",
                str(data_dir / "sft" / f"{slug}.jsonl"),
                "--out-dir",
                str(sft_data_dir),
            ],
        ),
    ]

    if dry_run:
        typer.echo(f"Pipeline plan for `{slug}`:")
        for stage, args in plan:
            done = "✓" if _stage_done(stage, status) else " "
            typer.echo(f"  [{done}] {stage}: civic-slm {' '.join(args)}")
        for stage in _STAGES[len(plan) :]:
            done = "✓" if _stage_done(stage, status) else " "
            typer.echo(f"  [{done}] {stage}")
        return

    for stage, args in plan:
        if _stage_done(stage, status):
            log.info("pipeline_skip", stage=stage)
            continue
        _run_cli(args)
        _mark(stage, status, slug, artifacts_dir)

    # train_cpt
    if not _stage_done("train_cpt", status):
        _write_stage_config(
            cpt_config_template,
            cpt_config_out,
            base_model=base_model,
            train_path=cpt_data_dir / "train.jsonl",
            valid_path=cpt_data_dir / "valid.jsonl",
            output_dir=cpt_adapter,
        )
        _run_cli(
            [
                "train",
                "cpt",
                "--config",
                str(cpt_config_out),
                "--max-iters",
                str(cpt_iters),
            ]
        )
        _mark("train_cpt", status, slug, artifacts_dir)

    # fuse_cpt
    if not _stage_done("fuse_cpt", status):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm.fuse",
                "--model",
                base_model,
                "--adapter-path",
                str(cpt_adapter),
                "--save-path",
                str(cpt_fused),
            ],
            check=True,
        )
        _mark("fuse_cpt", status, slug, artifacts_dir)

    # train_sft
    if not _stage_done("train_sft", status):
        _write_stage_config(
            sft_config_template,
            sft_config_out,
            base_model=str(cpt_fused),
            train_path=sft_data_dir / "train.jsonl",
            valid_path=sft_data_dir / "valid.jsonl",
            output_dir=sft_adapter,
        )
        _run_cli(["train", "sft", "--config", str(sft_config_out)])
        _mark("train_sft", status, slug, artifacts_dir)

    # fuse_v1
    if not _stage_done("fuse_v1", status):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm.fuse",
                "--model",
                str(cpt_fused),
                "--adapter-path",
                str(sft_adapter),
                "--save-path",
                str(v1_fused),
            ],
            check=True,
        )
        _mark("fuse_v1", status, slug, artifacts_dir)

    # quantize
    if not skip_quantize and not _stage_done("quantize", status):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm.convert",
                "--hf-path",
                str(v1_fused),
                "--mlx-path",
                str(v1_quant),
                "-q",
                "--q-bits",
                "4",
            ],
            check=True,
        )
        _mark("quantize", status, slug, artifacts_dir)

    # eval — run all three runner benches against the v1 model
    if not skip_eval and not _stage_done("eval", status):
        bench_files = {
            "factuality": "data/eval/civic_factuality.jsonl",
            "refusal": "data/eval/refusal.jsonl",
            "extraction": "data/eval/structured_extraction.jsonl",
        }
        # The pipeline trained on this slug's manifest, so eval examples
        # whose source docs overlap with that manifest are expected and
        # not a contamination failure.
        for bench, path in bench_files.items():
            _run_cli(
                [
                    "eval",
                    "run",
                    "--model",
                    f"{slug}-v1",
                    "--bench",
                    bench,
                    "--bench-file",
                    path,
                    "--max-tokens",
                    "1024",
                    "--allow-contamination",
                ]
            )
        _mark("eval", status, slug, artifacts_dir)

    typer.echo(
        f"\nDone. Artifacts under `artifacts/{slug}-v1-fused/` (and `-mlx-q4/` if not skipped)."
        f"\nResume status: {_status_path(slug, artifacts_dir)}"
    )


if __name__ == "__main__":
    typer.run(main)
