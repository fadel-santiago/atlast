from pathlib import Path

from atlast.index import VectorIndex


def test_duplicate_content_is_not_indexed_twice(tmp_path: Path, embedder) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "copy.txt"
    content = "alpha retrieval content"
    first_path.write_text(content, encoding="utf-8")
    second_path.write_text(content, encoding="utf-8")

    with VectorIndex(tmp_path / "qdrant", embedder, chunk_size=10, overlap=0) as index:
        first = index.ingest(first_path)
        duplicate = index.ingest(second_path)
        hits = index.search("alpha", limit=10)

    assert first.status == "indexed"
    assert first.chunks == 1
    assert duplicate.status == "duplicate"
    assert duplicate.document_id == first.document_id
    assert duplicate.chunks == 0
    assert len(hits) == 1


def test_source_metadata_and_index_persist_across_reopening(
    tmp_path: Path,
    embedder,
    synthetic_pdf: Path,
) -> None:
    db_path = tmp_path / "persistent-qdrant"
    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as index:
        result = index.ingest(synthetic_pdf)
        assert result.status == "indexed"
        assert result.chunks == 2

    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as reopened:
        hits = reopened.search("Beta second page", limit=2)

    assert hits
    beta_hit = next(hit for hit in hits if "Beta" in hit.text)
    assert beta_hit.metadata["source"] == str(synthetic_pdf.resolve())
    assert beta_hit.metadata["filename"] == "sample.pdf"
    assert beta_hit.metadata["file_type"] == "pdf"
    assert beta_hit.metadata["page"] == 2
    assert beta_hit.metadata["content_hash"] == result.content_hash


def test_text_document_remains_searchable_after_index_restart(tmp_path: Path, embedder) -> None:
    document = tmp_path / "persistent.txt"
    document.write_text(
        "The restart marker is cobalt and remains searchable after reopening.",
        encoding="utf-8",
    )
    db_path = tmp_path / "restart-qdrant"

    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as index:
        indexed = index.ingest(document)

    with VectorIndex(db_path, embedder, chunk_size=20, overlap=0) as reopened:
        hits = reopened.search("cobalt restart marker", limit=1)

    assert indexed.status == "indexed"
    assert len(hits) == 1
    assert "cobalt" in hits[0].text
    assert hits[0].metadata["document_id"] == indexed.document_id
