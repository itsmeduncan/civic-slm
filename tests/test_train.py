"""Tests for train command builders. Doesn't actually invoke MLX."""

from __future__ import annotations

from pathlib import Path

import pytest

from civic_slm.train.common import ConfigError, TrainConfig
from civic_slm.train.cpt import build_command as build_cpt
from civic_slm.train.dpo import build_command as build_dpo
from civic_slm.train.sft import build_command as build_sft


def test_cpt_command_writes_mlx_lm_yaml() -> None:
    """CPT builder materializes an mlx_lm.lora -c YAML and points the command at it.

    mlx_lm 0.31 took LoRA hyperparams off the CLI surface — `--lora-rank`,
    `--target-modules`, etc. are gone. The civic-slm builder now writes a
    YAML next to the adapter dir and invokes `mlx_lm.lora -c <yaml>`. This
    test asserts the YAML is on disk with rank/scale/iters matching the
    TrainConfig.
    """
    import yaml as _yaml

    cfg = TrainConfig.load(Path("configs/cpt.yaml"))
    cmd = build_cpt(cfg, iters_override=42)
    assert cmd[:2] == ["mlx_lm.lora", "-c"]
    yaml_path = Path(cmd[2])
    assert yaml_path.exists()
    payload = _yaml.safe_load(yaml_path.read_text())
    assert payload["model"] == cfg.base_model
    assert payload["fine_tune_type"] == "lora"
    assert payload["iters"] == 42
    assert payload["lora_parameters"]["rank"] == cfg.lora.rank
    # mlx_lm's `scale` is alpha/rank — civic-slm exposes alpha directly.
    assert payload["lora_parameters"]["scale"] == pytest.approx(cfg.lora.alpha / cfg.lora.rank)


def test_sft_command_writes_mlx_lm_yaml() -> None:
    import yaml as _yaml

    cfg = TrainConfig.load(Path("configs/sft.yaml"))
    cmd = build_sft(cfg, max_iters=100)
    assert cmd[:2] == ["mlx_lm.lora", "-c"]
    payload = _yaml.safe_load(Path(cmd[2]).read_text())
    assert payload["iters"] == 100
    assert payload["grad_checkpoint"] is True


def test_dpo_command_includes_beta() -> None:
    cfg = TrainConfig.load(Path("configs/dpo.yaml"))
    cmd = build_dpo(cfg, max_iters=50)
    assert "mlx_lm.dpo" in cmd
    assert "--beta" in cmd
    beta_idx = cmd.index("--beta")
    assert cmd[beta_idx + 1] == "0.1"


def test_unknown_yaml_key_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "stage: sft\n"
        "base_model: foo\n"
        "data: {train_path: x.jsonl, format: chat}\n"
        "lora: {rank: 32, alpha: 64, dropout: 0.05}\n"
        "train: {epochs: 3, max_seq_length: 4096, learning_rate: 2.0e-4}\n"
        "output_dir: artifacts/x\n"
        "typo_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        TrainConfig.load(bad)


def test_cpt_without_iters_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad-cpt.yaml"
    bad.write_text(
        "stage: cpt\n"
        "base_model: foo\n"
        "data: {train_path: x.jsonl, format: text}\n"
        "lora: {rank: 64, alpha: 128, dropout: 0.05}\n"
        "train: {batch_size: 1, max_seq_length: 2048, learning_rate: 1.0e-5}\n"
        "output_dir: artifacts/x\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="iters"):
        TrainConfig.load(bad)


def test_dpo_without_beta_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad-dpo.yaml"
    bad.write_text(
        "stage: dpo\n"
        "base_model: foo\n"
        "data: {train_path: x.jsonl, format: dpo}\n"
        "lora: {rank: 32, alpha: 64, dropout: 0.0}\n"
        "train: {epochs: 1, max_seq_length: 4096, learning_rate: 5.0e-7}\n"
        "output_dir: artifacts/x\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="beta"):
        TrainConfig.load(bad)
