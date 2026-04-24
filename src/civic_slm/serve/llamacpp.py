"""Helpers for serving a GGUF model via llama.cpp's `llama-server`.

Used primarily for the Qwen2.5-72B comparator in the side_by_side benchmark.
Run in a separate terminal:

    llama-server -m ~/models/qwen2.5-72b-instruct-q4_k_m.gguf -c 8192 --port 8081

Then point the eval runner at `http://localhost:8081`.
"""

from __future__ import annotations

import shlex
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:8081"


def serve_command(model_path: Path, port: int = 8081, ctx: int = 8192) -> str:
    return shlex.join(
        [
            "llama-server",
            "-m",
            str(model_path),
            "-c",
            str(ctx),
            "--port",
            str(port),
        ]
    )
