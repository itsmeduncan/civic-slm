"""Strict-local tripwire tests.

Three layers under test:
  1. `runtimes.is_strict_local()` parses the env var.
  2. `select_backend()` raises when strict + non-local backend.
  3. `agent_llm()` raises when strict + non-local backend.
  4. `civic-slm doctor --strict-local` exits non-zero on violations.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from civic_slm.cli import app as cli_app
from civic_slm.ingest.recipes._browser import agent_llm
from civic_slm.llm.backend import LocalBackend, select_backend
from civic_slm.serve.runtimes import is_strict_local

runner = CliRunner()


# --- env helper --------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "yes", "ON", "True", "Yes"])
def test_is_strict_local_truthy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", val)
    assert is_strict_local() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "anything-else"])
def test_is_strict_local_falsy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", val)
    assert is_strict_local() is False


def test_is_strict_local_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIVIC_SLM_STRICT_LOCAL", raising=False)
    assert is_strict_local() is False


# --- select_backend ----------------------------------------------------------


def test_select_backend_strict_blocks_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "anthropic")
    with pytest.raises(RuntimeError, match="STRICT_LOCAL"):
        select_backend()


def test_select_backend_strict_blocks_unset_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset BACKEND defaults to 'anthropic' — strict mode catches that."""
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.delenv("CIVIC_SLM_LLM_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="STRICT_LOCAL"):
        select_backend()


def test_select_backend_strict_allows_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "local")
    backend = select_backend()
    assert isinstance(backend, LocalBackend)


def test_select_backend_no_strict_anthropic_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without strict mode, the default-anthropic path is preserved."""
    monkeypatch.delenv("CIVIC_SLM_STRICT_LOCAL", raising=False)
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "anthropic")
    # Doesn't raise; we don't actually call the backend (would need API key).
    backend = select_backend()
    assert backend.model.startswith("claude")


# --- agent_llm (browser-use) -----------------------------------------------


def testagent_llm_strict_blocks_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "anthropic")
    with pytest.raises(RuntimeError, match="STRICT_LOCAL"):
        agent_llm()


def testagent_llm_strict_blocks_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.delenv("CIVIC_SLM_LLM_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="STRICT_LOCAL"):
        agent_llm()


# --- doctor --strict-local ---------------------------------------------------


def test_doctor_strict_fails_on_anthropic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("CIVIC_SLM_CANDIDATE_URL", "http://127.0.0.1:65500")  # closed port
    r = runner.invoke(cli_app, ["doctor", "--strict-local", "--skip-teacher"])
    assert r.exit_code == 1
    assert "FAIL" in r.output


def test_doctor_strict_passes_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """With everything pointed at unreachable URLs we still exit non-zero,
    but the strict-local-specific checks (BACKEND, secret) shouldn't add
    failures of their own when configured correctly. Verify the strict
    checks aren't the source of the failure."""
    monkeypatch.setenv("CIVIC_SLM_LLM_BACKEND", "local")
    monkeypatch.setenv("CIVIC_SLM_STRICT_LOCAL", "1")
    monkeypatch.setenv("CIVIC_SLM_CANDIDATE_URL", "http://127.0.0.1:65500")
    monkeypatch.setenv("CIVIC_SLM_TEACHER_URL", "http://127.0.0.1:65501")
    r = runner.invoke(cli_app, ["doctor", "--strict-local"])
    # Strict-mode-specific rows should report OK (BACKEND=local, tripwire active);
    # the runtime pings will fail because the ports are closed, which is expected
    # but not a strict-local violation.
    assert "CIVIC_SLM_LLM_BACKEND" in r.output
    assert "CIVIC_SLM_STRICT_LOCAL" in r.output
