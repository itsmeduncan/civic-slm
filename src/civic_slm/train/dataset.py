"""Shared helpers for computing training step counts from the dataset.

Why this exists: `mlx_lm.lora` and `mlx_lm.dpo` take `--iters`, not `--epochs`.
If you hand-wave iters from a constant ("1000 per epoch"), you'll end up
training hundredths of an epoch on a 50k-example dataset or 100 epochs on a
50-example one. The right answer is always `iters = ceil(epochs * N / batch)`
where N is the real line count of the training JSONL.
"""

from __future__ import annotations

import math
from pathlib import Path


def jsonl_line_count(path: Path) -> int:
    """Count non-empty lines in a JSONL file. Streams; doesn't load into memory."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def compute_iters(
    *,
    train_path: Path,
    epochs: int,
    batch_size: int,
    fallback: int = 500,
) -> int:
    """Return `ceil(epochs * N / batch_size)`, or `fallback` if the file is missing/empty.

    Missing file is not fatal here — dry-runs and tests must work without data
    on disk. The caller (training main) should separately guarantee the file
    exists before actually running.
    """
    n = jsonl_line_count(train_path)
    if n == 0:
        return fallback
    return math.ceil(epochs * n / max(1, batch_size))
