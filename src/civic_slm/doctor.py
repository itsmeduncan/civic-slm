"""`civic-slm doctor` — environment + runtime sanity check.

Runs in seconds, hits every dependency the pipeline needs, prints a single
green/yellow/red status block. The point: when the pipeline misbehaves, this
tells you whether the candidate URL is up, the teacher URL is up, your
secrets are loaded, and (if local) which runtime you're talking to.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx
import typer
from rich.console import Console
from rich.table import Table

from civic_slm import config
from civic_slm.serve import runtimes

console = Console()
app = typer.Typer(help="Diagnose civic-slm runtime + secret configuration.")

Status = Literal["ok", "warn", "fail", "skip"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    latency_ms: float = 0.0


def _color(status: Status) -> str:
    return {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}[status]


def _ping_chat(base_url: str, model: str) -> Check:
    """POST a 1-token chat to /v1/chat/completions; report status + latency."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.post(url, json=payload, headers={"Authorization": "Bearer not-needed"})
        latency = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            try:
                data = r.json()
                model_back = data.get("model", model)
                return Check(
                    name=base_url,
                    status="ok",
                    detail=f"served={model_back}",
                    latency_ms=latency,
                )
            except (KeyError, ValueError):
                return Check(
                    name=base_url,
                    status="warn",
                    detail="HTTP 200 but unexpected body shape",
                    latency_ms=latency,
                )
        return Check(
            name=base_url,
            status="fail",
            detail=f"HTTP {r.status_code}: {r.text[:100]}",
            latency_ms=latency,
        )
    except httpx.HTTPError as exc:
        return Check(name=base_url, status="fail", detail=f"connection failed: {exc}")


def _check_secret(name: str) -> Check:
    s = config.settings()
    val = getattr(s, name.lower(), None)
    if val:
        return Check(name=name, status="ok", detail="loaded")
    return Check(name=name, status="warn", detail="not set (optional unless this stage needs it)")


def _looks_local(base_url: str) -> bool:
    """Soft check: is the URL pointed at loopback or a private network?

    Used by `--strict-local` to flag suspicious endpoints. Heuristic only —
    Tailscale / ZeroTier / *.local mDNS / private DNS names look "non-local"
    here, so we soft-warn (never hard-fail) on a no-match. False positives
    are acceptable; a false-clean would silently bill paid endpoints.
    """
    import urllib.parse

    try:
        host = urllib.parse.urlparse(base_url).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if host.endswith(".local"):
        return True
    return host.startswith(("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19."))


@app.command()
def main(
    skip_teacher: bool = typer.Option(False, help="Don't ping the teacher URL."),
    strict_local: bool = typer.Option(
        False,
        "--strict-local",
        help=(
            "Audit for zero-API-spend operation: backend must be `local`, "
            "ANTHROPIC_API_KEY must not be loaded, teacher URL must respond, "
            "candidate URL should look local. Exits non-zero on any violation."
        ),
    ),
) -> None:
    """Run sanity checks against env, secrets, and configured endpoints."""
    checks: list[tuple[str, Check]] = []

    # Secrets
    anthropic_check = _check_secret("ANTHROPIC_API_KEY")
    if strict_local and anthropic_check.status == "ok":
        # Strict-local doesn't want this key loaded at all — even if BACKEND=local
        # would override it, leaving the key in the env is a footgun.
        anthropic_check = Check(
            name="ANTHROPIC_API_KEY",
            status="fail",
            detail="loaded but --strict-local forbids it (unset to remove the footgun)",
        )
    checks.append(("ANTHROPIC_API_KEY", anthropic_check))
    checks.append(("HF_TOKEN", _check_secret("HF_TOKEN")))
    checks.append(("WANDB_API_KEY", _check_secret("WANDB_API_KEY")))

    # Backend choice
    backend = os.environ.get("CIVIC_SLM_LLM_BACKEND", "anthropic")
    backend_status: Status = "ok"
    backend_detail = f"= {backend!r}"
    if strict_local and backend != "local":
        backend_status = "fail"
        backend_detail = f"= {backend!r} (--strict-local requires `local`)"
    checks.append(
        (
            "CIVIC_SLM_LLM_BACKEND",
            Check(name=backend, status=backend_status, detail=backend_detail),
        )
    )

    # Strict-local tripwire status
    if strict_local:
        tripwire_set = runtimes.is_strict_local()
        checks.append(
            (
                "CIVIC_SLM_STRICT_LOCAL",
                Check(
                    name="strict-local",
                    status="ok" if tripwire_set else "warn",
                    detail=(
                        "runtime tripwire active"
                        if tripwire_set
                        else "not set in env — guard only enforced for this `doctor` run"
                    ),
                ),
            )
        )

    # Candidate runtime ping (for eval)
    cand_url, cand_model = runtimes.candidate_url(), runtimes.candidate_model()
    cand_check = _ping_chat(cand_url, cand_model)
    if strict_local and cand_check.status == "ok" and not _looks_local(cand_url):
        cand_check = Check(
            name=cand_url,
            status="warn",
            detail=(
                f"{cand_check.detail} (URL doesn't look local — confirm it's not a paid endpoint)"
            ),
            latency_ms=cand_check.latency_ms,
        )
    checks.append(("candidate runtime", cand_check))

    # Teacher runtime ping (only if local backend is selected)
    if backend == "local" and not skip_teacher:
        t_url, t_model = runtimes.teacher_url(), runtimes.teacher_model()
        teacher_check = _ping_chat(t_url, t_model)
        if strict_local and teacher_check.status != "ok":
            teacher_check = Check(
                name=t_url,
                status="fail",
                detail=f"{teacher_check.detail} (--strict-local requires teacher reachable)",
                latency_ms=teacher_check.latency_ms,
            )
        elif strict_local and not _looks_local(t_url):
            teacher_check = Check(
                name=t_url,
                status="warn",
                detail=f"{teacher_check.detail} (URL doesn't look local)",
                latency_ms=teacher_check.latency_ms,
            )
        checks.append(("teacher runtime", teacher_check))
    elif backend == "anthropic":
        skip_status: Status = "fail" if strict_local else "skip"
        skip_detail = (
            "using Anthropic SDK — forbidden by --strict-local"
            if strict_local
            else "using Anthropic SDK"
        )
        checks.append(
            (
                "teacher runtime",
                Check(name="(anthropic API)", status=skip_status, detail=skip_detail),
            )
        )

    table = Table(title="civic-slm doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail")
    table.add_column("ms", justify="right")

    overall: Status = "ok"
    for name, c in checks:
        if c.status == "fail":
            overall = "fail"
        elif c.status == "warn" and overall != "fail":
            overall = "warn"
        table.add_row(
            name,
            f"[{_color(c.status)}]{c.status.upper()}[/]",
            c.detail,
            f"{c.latency_ms:.0f}" if c.latency_ms else "",
        )

    console.print(table)
    console.print(
        f"\nOverall: [{_color(overall)}]{overall.upper()}[/]\n"
        f"Candidate: {cand_url} (model={cand_model})"
    )
    if backend == "local":
        console.print(f"Teacher:   {runtimes.teacher_url()} (model={runtimes.teacher_model()})")
    if overall == "fail":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
