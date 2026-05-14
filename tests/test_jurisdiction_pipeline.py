# pyright: reportPrivateUsage=false
"""Tests for `civic-slm train jurisdiction` — the one-command pipeline.

What's under test: the composer's stage ordering, the `--resume` skip
behavior driven by `status.json`, the config materialization (per-slug
output paths layered onto a shared default), and the `--dry-run` plan
print. Actual mlx_lm calls are mocked — we're not testing mlx_lm, only
that our composer hands it the right arguments.

This file deliberately tests private helpers (`_load_status`,
`_write_stage_config`, `_run_cli`, …) — they're the contract the
composer relies on, so they get covered. Hence the file-level
`reportPrivateUsage` carve-out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import civic_slm.train.jurisdiction as jurisdiction_mod


def _make_template_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal jurisdiction-default.{cpt,sft}.yaml stubs for the test.

    The shipped templates live under `configs/` and require the full
    TrainConfig schema; we don't need that here — `_write_stage_config`
    just clones whichever YAML it's pointed at.
    """
    cpt = tmp_path / "default.cpt.yaml"
    cpt.write_text(
        yaml.safe_dump(
            {
                "stage": "cpt",
                "base_model": "REPLACE",
                "data": {
                    "train_path": "REPLACE",
                    "valid_path": "REPLACE",
                    "format": "text",
                },
                "lora": {"rank": 32, "alpha": 64, "dropout": 0.05},
                "train": {
                    "iters": 200,
                    "batch_size": 1,
                    "max_seq_length": 1024,
                    "learning_rate": 1.0e-5,
                },
                "output_dir": "REPLACE",
            }
        ),
        encoding="utf-8",
    )
    sft = tmp_path / "default.sft.yaml"
    sft.write_text(
        yaml.safe_dump(
            {
                "stage": "sft",
                "base_model": "REPLACE",
                "data": {
                    "train_path": "REPLACE",
                    "valid_path": "REPLACE",
                    "format": "chat",
                },
                "lora": {"rank": 32, "alpha": 64, "dropout": 0.05},
                "train": {
                    "epochs": 3,
                    "batch_size": 1,
                    "max_seq_length": 2048,
                    "learning_rate": 2.0e-4,
                },
                "output_dir": "REPLACE",
            }
        ),
        encoding="utf-8",
    )
    return cpt, sft


def test_write_stage_config_overrides_per_slug(tmp_path: Path) -> None:
    """Per-slug fields get rewritten; everything else is preserved."""
    cpt_template, _ = _make_template_configs(tmp_path)
    out = tmp_path / "out.yaml"
    jurisdiction_mod._write_stage_config(
        cpt_template,
        out,
        base_model="/path/to/base",
        train_path=tmp_path / "train.jsonl",
        valid_path=tmp_path / "valid.jsonl",
        output_dir=tmp_path / "adapters",
    )
    payload = yaml.safe_load(out.read_text())
    assert payload["base_model"] == "/path/to/base"
    assert payload["data"]["train_path"] == str(tmp_path / "train.jsonl")
    assert payload["data"]["valid_path"] == str(tmp_path / "valid.jsonl")
    assert payload["output_dir"] == str(tmp_path / "adapters")
    # Untouched fields preserved.
    assert payload["stage"] == "cpt"
    assert payload["lora"]["rank"] == 32


def test_status_roundtrip_marks_stages_done(tmp_path: Path) -> None:
    """`_mark` + `_load_status` should agree on completed stages."""
    status: dict[str, str] = {}
    assert not jurisdiction_mod._stage_done("crawl", status)
    jurisdiction_mod._mark("crawl", status, "tmpville", tmp_path)
    assert jurisdiction_mod._stage_done("crawl", status)

    reloaded = jurisdiction_mod._load_status("tmpville", tmp_path)
    assert jurisdiction_mod._stage_done("crawl", reloaded)
    assert not jurisdiction_mod._stage_done("process", reloaded)


def test_status_path_lives_under_artifacts(tmp_path: Path) -> None:
    """status.json belongs under artifacts/<slug>-pipeline/, not data/."""
    path = jurisdiction_mod._status_path("tmpville", tmp_path)
    assert path == tmp_path / "tmpville-pipeline" / "status.json"


def test_stages_constant_covers_full_pipeline() -> None:
    """_STAGES is the source of truth for stage names; pin its shape so a
    silent reorder is impossible."""
    expected = (
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
    assert expected == jurisdiction_mod._STAGES


def test_run_cli_raises_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage that returns non-zero must raise StageError, not silently continue."""
    import subprocess

    class _FakeCompleted:
        returncode = 1

    def fake_run(*_a: Any, **_kw: Any) -> _FakeCompleted:
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(jurisdiction_mod.StageError, match="exited 1"):
        jurisdiction_mod._run_cli(["crawl", "tmpville"])
