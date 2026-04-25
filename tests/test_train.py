"""Tests for train command builders. Doesn't actually invoke MLX."""

from __future__ import annotations

from pathlib import Path

import pytest

from civic_slm.train.common import ConfigError, TrainConfig
from civic_slm.train.cpt import build_command as build_cpt
from civic_slm.train.dpo import build_command as build_dpo
from civic_slm.train.sft import build_command as build_sft


def test_cpt_command_includes_base_model_and_lora_rank() -> None:
    cfg = TrainConfig.load(Path("configs/cpt.yaml"))
    cmd = build_cpt(cfg)
    assert "mlx_lm.lora" in cmd
    assert cfg.base_model in cmd
    assert "--lora-rank" in cmd
    rank_idx = cmd.index("--lora-rank")
    assert cmd[rank_idx + 1] == "64"


def test_sft_command_uses_chat_data_dir() -> None:
    cfg = TrainConfig.load(Path("configs/sft.yaml"))
    cmd = build_sft(cfg, max_iters=100)
    assert "--iters" in cmd
    iters_idx = cmd.index("--iters")
    assert cmd[iters_idx + 1] == "100"
    assert "--grad-checkpoint" in cmd


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
