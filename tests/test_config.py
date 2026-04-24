"""Smoke tests for config loader. Don't read real secrets."""

from __future__ import annotations

from pathlib import Path

import pytest

from civic_slm import config


def test_settings_loads_without_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    config.settings.cache_clear()
    monkeypatch.setattr(config, "ENV_PATH", Path("/tmp/civic-slm-nonexistent.env"))
    s = config.settings()
    assert s.hf_token is None
    assert s.anthropic_api_key is None


def test_require_raises_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    config.settings.cache_clear()
    monkeypatch.setattr(config, "ENV_PATH", Path("/tmp/civic-slm-nonexistent.env"))
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        config.require("HF_TOKEN")
