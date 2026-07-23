from __future__ import annotations

import pytest

from civic_slm.serve.rag.attachments import (
    DocText,
    UnsupportedDocType,
    extract_document,
)


def test_txt_extraction_roundtrips() -> None:
    out = extract_document("notes.txt", b"Council approved item 9.")
    assert isinstance(out, DocText)
    assert out.text == "Council approved item 9."
    assert out.pages == 1
    assert out.chars == len(out.text)
    assert out.truncated is False


def test_unsupported_type_rejected() -> None:
    with pytest.raises(UnsupportedDocType):
        extract_document("photo.png", b"\x89PNG")


def test_oversized_text_is_truncated() -> None:
    big = "x" * 40_000
    out = extract_document("big.md", big.encode())
    assert out.truncated is True
    assert out.chars <= 30_000
    assert out.text.endswith("[document truncated]")


class TestAttachmentsHTTP:
    """HTTP-level coverage for `POST /v1/attachments` via `build_app()`.

    Gated on fastapi/starlette (needed for `TestClient`), which live outside
    the base install — CI's lint/type job doesn't sync a `rag` extra, so this
    class must skip cleanly rather than error when fastapi isn't installed.
    An autouse fixture (not a bare `pytest.importorskip` in the class body)
    is required here: the latter would raise during module collection and
    skip the whole file, taking the pure-extraction tests above down with it.
    """

    @pytest.fixture(autouse=True)
    def _require_fastapi(self) -> None:
        pytest.importorskip("fastapi")

    @pytest.fixture
    def client(self):  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient  # type: ignore[import-not-found]

        from civic_slm.serve.rag.cli import build_app

        return TestClient(build_app())

    def test_txt_upload_returns_extracted_text(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/v1/attachments",
            files={"file": ("notes.txt", b"Council approved item 9.", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "Council approved item 9."

    def test_png_upload_is_rejected_as_unsupported_type(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/v1/attachments",
            files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert resp.status_code == 415

    def test_oversized_upload_is_rejected(self, client) -> None:  # type: ignore[no-untyped-def]
        big = b"x" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            "/v1/attachments",
            files={"file": ("big.txt", big, "text/plain")},
        )
        assert resp.status_code == 413

    def test_bogus_pdf_upload_is_rejected_as_extraction_error(self, client) -> None:  # type: ignore[no-untyped-def]
        # Gated on pypdf because it lives in the optional `ingest` extra,
        # which CI's lint/type job doesn't sync (see test_ingest.py for the
        # same idiom).
        pytest.importorskip("pypdf")
        resp = client.post(
            "/v1/attachments",
            files={"file": ("fake.pdf", b"not actually a pdf", "application/pdf")},
        )
        assert resp.status_code == 422
