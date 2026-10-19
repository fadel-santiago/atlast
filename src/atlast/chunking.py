"""Deterministic document chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from atlast.documents import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    point_id: str
    text: str
    metadata: dict[str, Any]


def chunk_document(
    document: LoadedDocument,
    *,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Split each page/section into stable word windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    chunk_index = 0

    for section in document.sections:
        words = section.text.split()
        section_chunk_index = 0
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                continue

            text = " ".join(window)
            metadata: dict[str, Any] = {
                "document_id": document.document_id,
                "content_hash": document.content_hash,
                "source": document.source,
                "filename": document.filename,
                "file_type": document.file_type,
                "chunk_index": chunk_index,
                "section_chunk_index": section_chunk_index,
                **section.metadata,
            }
            point_id = str(
                uuid5(NAMESPACE_URL, f"atlast:{document.document_id}:{chunk_index}:{text}")
            )
            chunks.append(Chunk(point_id=point_id, text=text, metadata=metadata))
            chunk_index += 1
            section_chunk_index += 1

            if start + chunk_size >= len(words):
                break

    return chunks
