from pathlib import Path

import pytest

from atlast.documents import load_document
from atlast.errors import DocumentReadError, EmptyDocumentError


def test_load_txt_tracks_sections_identity_and_source(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("First section.\fSecond section.", encoding="utf-8")

    document = load_document(path)

    assert document.document_id == document.content_hash
    assert len(document.content_hash) == 64
    assert document.source == str(path.resolve())
    assert document.filename == "notes.txt"
    assert document.file_type == "txt"
    assert [section.metadata for section in document.sections] == [
        {"section": 1},
        {"section": 2},
    ]


def test_load_pdf_extracts_text_and_one_based_page_metadata(synthetic_pdf: Path) -> None:
    document = load_document(synthetic_pdf)

    assert [section.metadata for section in document.sections] == [{"page": 1}, {"page": 2}]
    assert "Alpha" in document.sections[0].text
    assert "Beta" in document.sections[1].text


@pytest.mark.parametrize("content", [b"", b" \n\t\f "])
def test_empty_txt_is_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(content)

    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_malformed_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a valid PDF")

    with pytest.raises(DocumentReadError):
        load_document(path)


def test_invalid_utf8_txt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(DocumentReadError):
        load_document(path)
