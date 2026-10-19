from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from atlast.errors import OllamaModelUnavailableError
from atlast.index import VectorIndex
from atlast.web import WebConfig, create_app


class WebFakeGenerator:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def generate(self, _prompt: str) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_client(tmp_path: Path, embedder, generated: str | Exception, *, limit: int = 1024):
    index = VectorIndex(tmp_path / "qdrant", embedder, chunk_size=30, overlap=0)
    config = WebConfig(
        db_path=tmp_path / "qdrant",
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=limit,
    )
    app = create_app(config, index=index, generator=WebFakeGenerator(generated))
    return TestClient(app), index, config


def test_health_and_interface_assets(tmp_path: Path, embedder) -> None:
    client, index, _ = make_client(tmp_path, embedder, "{}")
    try:
        with client:
            health = client.get("/api/health")
            page = client.get("/")
            script = client.get("/app.js")
    finally:
        index.close()

    assert health.status_code == 200
    assert health.json()["local_only"] is True
    assert health.json()["ollama_model"] == "llama3.2:3b"
    assert 'id="file-input"' in page.text
    assert 'id="ask-form"' in page.text
    assert 'id="citation-list"' in page.text
    assert 'id="document-list"' in page.text
    assert 'id="system-flow"' in page.text
    assert 'data-pipeline="ingestion"' in page.text
    assert 'data-pipeline="qa"' in page.text
    assert 'requestJson("/api/documents"' in script.text
    assert 'method: "DELETE"' in script.text
    assert 'method: "PUT"' in script.text
    assert "window.confirm" in script.text
    assert 'requestJson("/api/ask"' in script.text


def test_upload_sanitizes_filename_and_detects_duplicate(tmp_path: Path, embedder) -> None:
    client, index, config = make_client(tmp_path, embedder, "{}")
    content = b"ATLAST stores deterministic searchable passages."
    try:
        with client:
            first = client.post(
                "/api/documents",
                files={"file": ("../../unsafe report.txt", content, "text/plain")},
            )
            duplicate = client.post(
                "/api/documents",
                files={"file": ("copy.txt", content, "text/plain")},
            )
    finally:
        index.close()

    assert first.status_code == 200
    assert first.json()["status"] == "indexed"
    assert first.json()["filename"] == "unsafe-report.txt"
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate"
    assert duplicate.json()["source"] is None
    assert [path.name for path in config.upload_dir.iterdir()] == ["unsafe-report.txt"]


@pytest.mark.parametrize(
    ("filename", "content", "expected_status", "expected_code"),
    [
        ("empty.txt", b"", 422, "empty_document"),
        ("broken.pdf", b"%PDF-1.7 not a PDF", 422, "invalid_document"),
        ("image.png", b"image", 415, "unsupported_type"),
        ("large.txt", b"x" * 1025, 413, "upload_too_large"),
    ],
)
def test_upload_errors_are_safe_and_structured(
    tmp_path: Path,
    embedder,
    filename: str,
    content: bytes,
    expected_status: int,
    expected_code: str,
) -> None:
    client, index, config = make_client(tmp_path, embedder, "{}", limit=1024)
    try:
        with client:
            response = client.post(
                "/api/documents",
                files={"file": (filename, content, "application/octet-stream")},
            )
    finally:
        index.close()

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert list(config.upload_dir.iterdir()) == []


def test_upload_ask_and_retrieve_workflow(tmp_path: Path, embedder) -> None:
    generated = json.dumps(
        {
            "answer": "Deterministic indexing makes retrieval reproducible.",
            "supported": True,
            "citation_ids": ["S1"],
        }
    )
    client, index, _ = make_client(tmp_path, embedder, generated)
    try:
        with client:
            upload = client.post(
                "/api/documents",
                files={
                    "file": (
                        "guide.txt",
                        b"Deterministic indexing makes retrieval reproducible.",
                        "text/plain",
                    )
                },
            )
            answer = client.post(
                "/api/ask",
                json={"question": "What makes retrieval reproducible?", "limit": 1},
            )
            retrieval = client.post(
                "/api/retrieve",
                json={"query": "reproducible retrieval", "limit": 1},
            )
    finally:
        index.close()

    assert upload.status_code == 200
    assert answer.status_code == 200
    assert answer.json()["status"] == "answered"
    assert answer.json()["citation_ids"] == ["S1"]
    assert answer.json()["sources"][0]["document_title"] == "guide.txt"
    assert answer.json()["sources"][0]["section"] == 1
    assert answer.json()["trace"]["model"] == "llama3.2:3b"
    assert answer.json()["trace"]["context_chars"] > 0
    assert answer.json()["trace"]["retrieved_sources"][0]["citation_id"] == "S1"
    assert answer.json()["trace"]["retrieved_sources"][0]["retrieval_score"] == answer.json()["sources"][0]["retrieval_score"]
    assert retrieval.status_code == 200
    assert retrieval.json()["passages"][0]["passage_id"] == "R1"
    assert retrieval.json()["passages"][0]["document_title"] == "guide.txt"


def test_document_list_delete_and_replacement_workflow(tmp_path: Path, embedder) -> None:
    client, index, _ = make_client(tmp_path, embedder, "{}")
    try:
        with client:
            first = client.post(
                "/api/documents",
                files={"file": ("manual.txt", b"legacy amber procedure", "text/plain")},
            ).json()
            second = client.post(
                "/api/documents",
                files={"file": ("policy.txt", b"retained cobalt policy", "text/plain")},
            ).json()

            listing = client.get("/api/documents")
            failed = client.put(
                f"/api/documents/{first['document_id']}",
                files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
            )
            after_failure = client.get("/api/documents").json()
            replaced = client.put(
                f"/api/documents/{first['document_id']}",
                files={
                    "file": (
                        "manual-v2.txt",
                        b"current violet replacement procedure",
                        "text/plain",
                    )
                },
            )
            retrieval = client.post(
                "/api/retrieve",
                json={"query": "violet replacement procedure", "limit": 5},
            )
            deleted = client.delete(f"/api/documents/{second['document_id']}")
            final_listing = client.get("/api/documents")
    finally:
        index.close()

    assert listing.status_code == 200
    assert listing.json()["count"] == 2
    assert {item["document_id"] for item in listing.json()["documents"]} == {
        first["document_id"],
        second["document_id"],
    }
    assert failed.status_code == 422
    assert failed.json()["detail"]["code"] == "invalid_document"
    assert first["document_id"] in {
        item["document_id"] for item in after_failure["documents"]
    }
    assert replaced.status_code == 200
    assert replaced.json()["status"] == "replaced"
    assert replaced.json()["document_id"] != first["document_id"]
    assert retrieval.status_code == 200
    assert replaced.json()["document_id"] in {
        passage["document_id"] for passage in retrieval.json()["passages"]
    }
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunks"] == 1
    assert final_listing.json()["count"] == 1
    assert final_listing.json()["documents"][0]["document_id"] == replaced.json()[
        "document_id"
    ]


def test_replacement_with_identical_content_keeps_original(tmp_path: Path, embedder) -> None:
    client, index, _ = make_client(tmp_path, embedder, "{}")
    content = b"identical replacement content"
    try:
        with client:
            original = client.post(
                "/api/documents",
                files={"file": ("original.txt", content, "text/plain")},
            ).json()
            replacement = client.put(
                f"/api/documents/{original['document_id']}",
                files={"file": ("copy.txt", content, "text/plain")},
            )
            listing = client.get("/api/documents").json()
    finally:
        index.close()

    assert replacement.status_code == 200
    assert replacement.json()["status"] == "duplicate"
    assert replacement.json()["document_id"] == original["document_id"]
    assert listing["count"] == 1
    assert listing["documents"][0]["filename"] == "original.txt"


def test_retrieval_rejects_whitespace_query(tmp_path: Path, embedder) -> None:
    client, index, _ = make_client(tmp_path, embedder, "{}")
    try:
        with client:
            response = client.post("/api/retrieve", json={"query": "   "})
    finally:
        index.close()

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "empty_query"


def test_model_failure_is_an_understandable_api_error(tmp_path: Path, embedder) -> None:
    client, index, _ = make_client(
        tmp_path,
        embedder,
        OllamaModelUnavailableError("Requested Ollama model is not installed"),
    )
    document = tmp_path / "source.txt"
    document.write_text("Local evidence exists.", encoding="utf-8")
    index.ingest(document)
    try:
        with client:
            response = client.post("/api/ask", json={"question": "What exists?"})
    finally:
        index.close()

    assert response.status_code == 503
    assert response.json()["status"] == "model_unavailable"
    assert response.json()["citation_ids"] == []
