"""`civic-slm merge` — fuse the final adapter and emit MLX 4-bit + GGUF Q5_K_M.

Outputs (under `<out_root>/`):
  - civic-slm-v{version}-mlx-q4/                          MLX 4-bit (Apple Silicon native)
  - civic-slm-v{version}-gguf-q5km/civic-slm-v{version}-q5_k_m.gguf   llama.cpp / Ollama

Why two artifacts: MLX is the lowest-overhead inference path on Apple Silicon;
GGUF is what Ollama/llama.cpp/LM Studio expect for non-Mac runtimes. Shipping
both doubles disk but lets the OSS release cover both audiences.

Steps:
  1. mlx_lm.fuse        — merge LoRA adapter into base, save as MLX (no quant).
  2. mlx_lm.convert     — quantize the fused MLX checkpoint to 4-bit.
  3. llama.cpp convert  — convert the fused HF-style weights to F16 GGUF.
  4. llama-quantize     — quantize F16 GGUF to Q5_K_M.

Steps 3-4 require llama.cpp built locally (`brew install llama.cpp`).
The command surfaces a clear error if the binaries aren't on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer


def _need(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        typer.echo(
            f"{binary!r} not found on PATH. "
            f"Install it before running merge "
            f"(brew install llama.cpp; pip install mlx-lm).",
            err=True,
        )
        raise typer.Exit(code=2)
    return path


def fuse_mlx(adapter_dir: Path, base_model: str, fused_dir: Path) -> None:
    _need("mlx_lm.fuse")
    fused_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "mlx_lm.fuse",
            "--model",
            base_model,
            "--adapter-path",
            str(adapter_dir),
            "--save-path",
            str(fused_dir),
        ],
        check=True,
    )


def quantize_mlx_q4(fused_dir: Path, out_dir: Path) -> None:
    _need("mlx_lm.convert")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "mlx_lm.convert",
            "--hf-path",
            str(fused_dir),
            "--mlx-path",
            str(out_dir),
            "-q",
            "--q-bits",
            "4",
        ],
        check=True,
    )


def to_gguf_q5km(fused_dir: Path, out_dir: Path, name: str) -> Path:
    convert_bin = shutil.which("convert_hf_to_gguf.py") or shutil.which("llama-convert-hf-to-gguf")
    quantize_bin = _need("llama-quantize")
    if not convert_bin:
        typer.echo(
            "llama.cpp convert script not found. Build llama.cpp locally and ensure "
            "convert_hf_to_gguf.py is on PATH, or `brew install llama.cpp`.",
            err=True,
        )
        raise typer.Exit(code=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{name}-f16.gguf"
    q5 = out_dir / f"{name}-q5_k_m.gguf"
    subprocess.run(
        [convert_bin, str(fused_dir), "--outtype", "f16", "--outfile", str(f16)],
        check=True,
    )
    subprocess.run([quantize_bin, str(f16), str(q5), "Q5_K_M"], check=True)
    f16.unlink(missing_ok=True)
    return q5


def main(
    adapter_dir: Path = typer.Option(
        ..., help="Adapter dir from the final training stage (CPT/SFT/DPO output)."
    ),
    base_model: str = typer.Option(..., help="HF or MLX base model id."),
    version: str = typer.Option(..., help="Release tag, e.g. `v1`."),
    out_root: Path = typer.Option(Path("artifacts"), help="Artifact root."),
    skip_gguf: bool = typer.Option(False, help="Skip the GGUF conversion."),
    name_prefix: str = typer.Option(
        "civic-slm", help="Artifact filename prefix. Override if releasing under a different name."
    ),
) -> None:
    """Fuse + quantize a fine-tuned adapter into release artifacts."""
    fused = out_root / f"{name_prefix}-{version}-fused"
    mlx_q4 = out_root / f"{name_prefix}-{version}-mlx-q4"
    gguf_dir = out_root / f"{name_prefix}-{version}-gguf-q5km"

    fuse_mlx(adapter_dir, base_model, fused)
    quantize_mlx_q4(fused, mlx_q4)
    typer.echo(f"MLX 4-bit: {mlx_q4}")
    if not skip_gguf:
        gguf_path = to_gguf_q5km(fused, gguf_dir, f"{name_prefix}-{version}")
        typer.echo(f"GGUF Q5_K_M: {gguf_path}")


if __name__ == "__main__":
    typer.run(main)
