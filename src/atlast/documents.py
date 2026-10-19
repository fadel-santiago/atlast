"""Document loading with stable identities and source metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from atlast.errors import DocumentReadError, EmptyDocumentError


@dataclass(frozen=True)
class DocumentSection:
    """A page or logical text section extracted from one source."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDocument:
    """Extracted document with a content-derived stable identity."""

    document_id: str
    content_hash: str
    source: str
    filename: str
    file_type: str
    sections: tuple[DocumentSection, ...]


def load_document(path: str | Path) -> LoadedDocument:
    """Load a UTF-8 TXT or PDF file and retain page/section boundaries."""

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise DocumentReadError(f"Document does not exist or is not a file: {source_path}")

    file_type = source_path.suffix.lower().lstrip(".")
    if file_type not in {"pdf", "txt"}:
        raise DocumentReadError(
            f"Unsupported document type '.{file_type or '(none)'}'; expected .pdf or .txt"
        )

    try:
        raw_content = source_path.read_bytes()
    except OSError as exc:
        raise DocumentReadError(f"Could not read document: {source_path}") from exc

    if not raw_content:
        raise EmptyDocumentError(f"Document is empty: {source_path}")

    if file_type == "txt":
        sections = _load_txt(raw_content, source_path)
    else:
        sections = _load_pdf(raw_content, source_path)

    if not sections:
        raise EmptyDocumentError(f"Document contains no extractable text: {source_path}")

    content_hash = hashlib.sha256(raw_content).hexdigest()
    return LoadedDocument(
        document_id=content_hash,
        content_hash=content_hash,
        source=str(source_path.resolve()),
        filename=source_path.name,
        file_type=file_type,
        sections=tuple(sections),
    )


def _load_txt(raw_content: bytes, path: Path) -> list[DocumentSection]:
    try:
        text = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentReadError(f"TXT document is not valid UTF-8: {path}") from exc

    sections: list[DocumentSection] = []
    for section_number, section_text in enumerate(text.split("\f"), start=1):
        if cleaned := section_text.strip():
            sections.append(
                DocumentSection(text=cleaned, metadata={"section": section_number})
            )
    return sections


def _load_pdf(raw_content: bytes, path: Path) -> list[DocumentSection]:
    try:
        reader = PdfReader(BytesIO(raw_content), strict=True)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise DocumentReadError(f"PDF is encrypted and cannot be read: {path}")

        sections: list[DocumentSection] = []
        for page_number, page in enumerate(reader.pages, start=1):
            if text := (page.extract_text() or "").strip():
                sections.append(DocumentSection(text=text, metadata={"page": page_number}))
        return sections
    except DocumentReadError:
        raise
    except Exception as exc:
        raise DocumentReadError(f"Could not parse PDF document: {path}") from exc
