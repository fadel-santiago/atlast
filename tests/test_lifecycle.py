from pathlib import Path

import pytest

from atlast.errors import DocumentNotFoundError
from atlast.index import VectorIndex


def test_list_documents_reports_stable_ids_chunks_and_metadata(
    tmp_path: Path,
    embedder,
    synthetic_pdf: Path,
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("first section\fsecond section", encoding="utf-8")

    with VectorIndex(tmp_path / "qdrant", embedder, chunk_size=20, overlap=0) as index:
        pdf_result = index.ingest(synthetic_pdf)
        text_result = index.ingest(text)
        documents = index.list_documents()

    by_id = {document.document_id: document for document in documents}
    assert set(by_id) == {pdf_result.document_id, text_result.document_id}
    assert by_id[pdf_result.document_id].filename == "sample.pdf"
    assert by_id[pdf_result.document_id].chunk_count == 2
    assert by_id[pdf_result.document_id].pages == (1, 2)
    assert by_id[text_result.document_id].source == str(text.resolve())
    assert by_id[text_result.document_id].file_type == "txt"
    assert by_id[text_result.document_id].sections == (1, 2)


def test_delete_document_is_isolated_and_cannot_delete_without_an_id(
    tmp_path: Path,
    embedder,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("amber lifecycle marker", encoding="utf-8")
    second.write_text("cobalt retained marker", encoding="utf-8")

    with VectorIndex(tmp_path / "qdrant", embedder, chunk_size=20, overlap=0) as index:
        first_result = index.ingest(first)
        second_result = index.ingest(second)
        deleted_chunks = index.delete_document(first_result.document_id)

        assert deleted_chunks == 1
        assert [item.document_id for item in index.list_documents()] == [
            second_result.document_id
        ]
        hits = index.search("cobalt retained marker", limit=10)
        assert hits
        assert {hit.metadata["document_id"] for hit in hits} == {
            second_result.document_id
        }
        with pytest.raises(DocumentNotFoundError):
            index.delete_document(first_result.document_id)
        with pytest.raises(ValueError):
            index.delete_document("   ")
        assert len(index.list_documents()) == 1


def test_failed_replacement_preserves_existing_document(
    tmp_path: Path,
    embedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tmp_path / "original.txt"
    replacement = tmp_path / "replacement.txt"
    original.write_text("original evidence remains available", encoding="utf-8")
    replacement.write_text("new evidence must not replace on failure", encoding="utf-8")

    with VectorIndex(tmp_path / "qdrant", embedder, chunk_size=20, overlap=0) as index:
        indexed = index.ingest(original)

        def fail_upsert(*_args, **_kwargs) -> None:
            raise RuntimeError("simulated storage failure")

        monkeypatch.setattr(index._client, "upsert", fail_upsert)
        with pytest.raises(RuntimeError, match="simulated storage failure"):
            index.replace_document(indexed.document_id, replacement)

        documents = index.list_documents()
        hits = index.search("original evidence", limit=1)

    assert [document.document_id for document in documents] == [indexed.document_id]
    assert hits[0].metadata["document_id"] == indexed.document_id


def test_replacement_and_retrieval_persist_after_restart(
    tmp_path: Path,
    embedder,
) -> None:
    original = tmp_path / "manual.txt"
    replacement = tmp_path / "manual-v2.txt"
    original.write_text("legacy amber procedure", encoding="utf-8")
    replacement.write_text("current violet replacement procedure", encoding="utf-8")
    db_path = tmp_path / "persistent-qdrant"

    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as index:
        old = index.ingest(original)
        replaced = index.replace_document(old.document_id, replacement)
        assert replaced.status == "replaced"
        assert replaced.document_id != old.document_id
        assert {item.document_id for item in index.list_documents()} == {
            replaced.document_id
        }

    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as reopened:
        documents = reopened.list_documents()
        hits = reopened.search("violet replacement procedure", limit=5)

    assert [document.document_id for document in documents] == [replaced.document_id]
    assert hits
    assert {hit.metadata["document_id"] for hit in hits} == {replaced.document_id}
    assert "violet" in hits[0].text
