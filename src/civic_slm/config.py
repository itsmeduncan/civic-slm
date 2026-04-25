"""Loads secrets and project paths from `~/.config/civic-slm/.env`.

Why a dedicated location instead of a project-local `.env`: the secrets follow the
user, not the working tree. Keeps tokens out of any worktree, branch, or container
mount that might leak them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path.home() / ".config" / "civic-slm"
ENV_PATH = CONFIG_DIR / ".env"


def _find_project_root() -> Path:
    """Walk up from this file looking for the repo root marker.

    We prefer markers in this order: `.git`, `pyproject.toml`, `VERSION`. This
    keeps `project_root` correct in the editable install (repo tree) AND
    falls through sensibly when installed as a wheel (no markers found →
    fall back to the CWD, which is what users expect when they invoke
    `civic-slm` from a project checkout).
    """
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")

    project_root: Path = Field(default_factory=_find_project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Return process-wide settings loaded from `~/.config/civic-slm/.env`.

    Missing file is fine — secrets become None, which is fine for dev/test.
    Stages that need a particular secret should assert on it themselves with a
    clear message.
    """
    raw: dict[str, str | None] = {}
    if ENV_PATH.exists():
        raw = dict(dotenv_values(ENV_PATH))
    return Settings.model_validate(raw)


def require(name: str) -> str:
    """Fetch a required secret or raise with an actionable message."""
    s = settings()
    value = getattr(s, name.lower(), None)
    if not value:
        raise RuntimeError(
            f"Missing secret {name!r}. Add it to {ENV_PATH} as `{name}=...` and retry.",
        )
    return value
