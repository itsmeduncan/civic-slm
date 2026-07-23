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
