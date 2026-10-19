"""Persistent Qdrant-backed document index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from atlast.chunking import chunk_document
from atlast.documents import LoadedDocument, load_document
from atlast.embeddings import Embedder
from atlast.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    IndexConfigurationError,
)


DEFAULT_COLLECTION = "documents"


@dataclass(frozen=True)
class IngestResult:
    status: str
    document_id: str
    content_hash: str
    source: str
    chunks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    score: float
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    content_hash: str
    filename: str
    source: str
    file_type: str
    chunk_count: int
    pages: tuple[int, ...]
    sections: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pages"] = list(self.pages)
        payload["sections"] = list(self.sections)
        return payload


@dataclass(frozen=True)
class ReplacementResult:
    status: str
    previous_document_id: str
    document_id: str
    content_hash: str
    source: str
    chunks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VectorIndex:
    """Own a local Qdrant client and index documents into one collection."""

    def __init__(
        self,
        db_path: str | Path,
        embedder: Embedder,
        *,
        collection: str = DEFAULT_COLLECTION,
        chunk_size: int = 200,
        overlap: int = 40,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.collection = collection
        self.chunk_size = chunk_size
        self.overlap = overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.db_path))
        self._dimension = self.embedder.dimension
        try:
            self._ensure_collection()
        except Exception:
            self._client.close()
            raise

    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def ingest(self, path: str | Path) -> IngestResult:
        document = load_document(path)
        if self.contains_hash(document.content_hash):
            return IngestResult(
                status="duplicate",
                document_id=document.document_id,
                content_hash=document.content_hash,
                source=document.source,
                chunks=0,
            )

        return self._index_document(document)

    def list_documents(self) -> list[IndexedDocument]:
        grouped: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                document_id = str(payload.get("document_id") or "").strip()
                if not document_id:
                    continue
                entry = grouped.setdefault(
                    document_id,
                    {
                        "document_id": document_id,
                        "content_hash": str(payload.get("content_hash") or document_id),
                        "filename": str(payload.get("filename") or "Unknown document"),
                        "source": str(payload.get("source") or ""),
                        "file_type": str(payload.get("file_type") or ""),
                        "chunk_count": 0,
                        "pages": set(),
                        "sections": set(),
                    },
                )
                entry["chunk_count"] += 1
                if isinstance(payload.get("page"), int):
                    entry["pages"].add(payload["page"])
                if isinstance(payload.get("section"), int):
                    entry["sections"].add(payload["section"])
            if offset is None:
                break

        documents = [
            IndexedDocument(
                document_id=entry["document_id"],
                content_hash=entry["content_hash"],
                filename=entry["filename"],
                source=entry["source"],
                file_type=entry["file_type"],
                chunk_count=entry["chunk_count"],
                pages=tuple(sorted(entry["pages"])),
                sections=tuple(sorted(entry["sections"])),
            )
            for entry in grouped.values()
        ]
        return sorted(documents, key=lambda item: (item.filename.lower(), item.document_id))

    def delete_document(self, document_id: str) -> int:
        document_id = self._validated_document_id(document_id)
        document_filter = self._document_filter(document_id)
        count = self._client.count(
            collection_name=self.collection,
            count_filter=document_filter,
            exact=True,
        ).count
        if count == 0:
            raise DocumentNotFoundError(f"Indexed document not found: {document_id}")

        self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=document_filter),
            wait=True,
        )
        return int(count)

    def replace_document(self, document_id: str, path: str | Path) -> ReplacementResult:
        document_id = self._validated_document_id(document_id)
        if not self._document_exists(document_id):
            raise DocumentNotFoundError(f"Indexed document not found: {document_id}")

        replacement = load_document(path)
        if replacement.document_id == document_id:
            return ReplacementResult(
                status="duplicate",
                previous_document_id=document_id,
                document_id=document_id,
                content_hash=replacement.content_hash,
                source=replacement.source,
                chunks=0,
            )
        if self.contains_hash(replacement.content_hash):
            raise DuplicateDocumentError(
                "Replacement content is already indexed as another document."
            )

        indexed = self._index_document(replacement)
        try:
            self.delete_document(document_id)
        except Exception:
            try:
                self.delete_document(indexed.document_id)
            except Exception:
                pass
            raise
        return ReplacementResult(
            status="replaced",
            previous_document_id=document_id,
            document_id=indexed.document_id,
            content_hash=indexed.content_hash,
            source=indexed.source,
            chunks=indexed.chunks,
        )

    def _index_document(self, document: LoadedDocument) -> IngestResult:
        chunks = chunk_document(
            document,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        if not chunks:
            raise EmptyDocumentError(f"Document produced no chunks: {document.source}")

        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise IndexConfigurationError("Embedder returned an unexpected vector count")
        if any(len(vector) != self._dimension for vector in vectors):
            raise IndexConfigurationError("Embedder returned vectors with an unexpected size")

        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "embedding_model": self.embedder.name,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )
        return IngestResult(
            status="indexed",
            document_id=document.document_id,
            content_hash=document.content_hash,
            source=document.source,
            chunks=len(chunks),
        )

    def _document_exists(self, document_id: str) -> bool:
        points, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=self._document_filter(document_id),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    @staticmethod
    def _validated_document_id(document_id: str) -> str:
        validated = document_id.strip()
        if not validated:
            raise ValueError("document_id must not be empty")
        return validated

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    def contains_hash(self, content_hash: str) -> bool:
        points, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_vector = self.embedder.encode([query])[0]
        if len(query_vector) != self._dimension:
            raise IndexConfigurationError("Embedder returned a query vector with an unexpected size")

        response = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            hits.append(SearchHit(score=float(point.score), text=text, metadata=payload))
        return hits

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection_info = self._client.get_collection(self.collection)
        vectors_config = collection_info.config.params.vectors
        existing_size = getattr(vectors_config, "size", None)
        if existing_size is not None and int(existing_size) != self._dimension:
            raise IndexConfigurationError(
                f"Collection '{self.collection}' uses vector size {existing_size}, "
                f"but embedder '{self.embedder.name}' uses {self._dimension}"
            )
