"""Tests for train command builders. Doesn't actually invoke MLX."""

from __future__ import annotations

from pathlib import Path

from civic_slm.train.common import TrainConfig
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
