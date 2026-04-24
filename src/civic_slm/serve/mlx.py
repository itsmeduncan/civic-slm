"""Helpers for serving an MLX model locally via `mlx_lm.server`.

Usage (in a separate terminal):

    mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

Then point the eval runner at `http://localhost:8080` with `--model qwen2.5-7b`.
"""

from __future__ import annotations

import shlex

DEFAULT_BASE_URL = "http://localhost:8080"


def serve_command(model: str, port: int = 8080) -> str:
    """Return the shell command to start an MLX-LM server for `model`."""
    return shlex.join(["mlx_lm.server", "--model", model, "--port", str(port)])
