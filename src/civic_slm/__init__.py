"""Civic SLM — domain-specialized fine-tune of Qwen2.5-7B for U.S. local-government documents."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("civic-slm")
except PackageNotFoundError:  # editable / source checkout without metadata
    from pathlib import Path

    _version_file = Path(__file__).resolve().parents[2] / "VERSION"
    __version__ = _version_file.read_text().strip() if _version_file.exists() else "0.0.0+unknown"
