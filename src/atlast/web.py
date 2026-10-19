"""Local-only FastAPI workflow for ATLAST ingestion, retrieval, and QA."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, AsyncIterator
import unicodedata
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from atlast.embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from atlast.errors import (
    AtlastError,
    DocumentNotFoundError,
    DocumentReadError,
    DuplicateDocumentError,
    EmptyDocumentError,
)
from atlast.index import DEFAULT_COLLECTION, SearchHit, VectorIndex
from atlast.qa import (
    DEFAULT_MAX_CONTEXT_CHARS,
    OllamaGenerator,
    QuestionAnsweringService,
    TextGenerator,
)


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt"}


@dataclass(frozen=True)
class WebConfig:
    db_path: Path = field(
        default_factory=lambda: Path(os.environ.get("ATLAST_DB_PATH", ".atlast/qdrant"))
    )
    upload_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ATLAST_UPLOAD_DIR", ".atlast/uploads"))
    )
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_MODEL
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    )
    ollama_url: str = field(
        default_factory=lambda: os.environ.get(
            "ATLAST_OLLAMA_URL", "http://127.0.0.1:11434"
        )
    )
    ollama_timeout: float = 180.0
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    limit: int = Field(default=4, ge=1, le=10)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


def create_app(
    config: WebConfig | None = None,
    *,
    index: VectorIndex | None = None,
    generator: TextGenerator | None = None,
) -> FastAPI:
    """Create an app, optionally injecting local test doubles."""

    web_config = config or WebConfig()
    if web_config.max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be greater than zero")
    if web_config.max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")
    owns_index = index is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        web_config.upload_dir.mkdir(parents=True, exist_ok=True)
        app.state.index = index or await run_in_threadpool(
            lambda: VectorIndex(
                web_config.db_path,
                SentenceTransformerEmbedder(web_config.embedding_model),
                collection=web_config.collection,
            )
        )
        app.state.generator = generator or OllamaGenerator(
            web_config.ollama_model,
            base_url=web_config.ollama_url,
            timeout=web_config.ollama_timeout,
        )
        app.state.operation_lock = asyncio.Lock()
        try:
            yield
        finally:
            if owns_index:
                await run_in_threadpool(app.state.index.close)

    app = FastAPI(
        title="ATLAST Local API",
        version="0.5.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = web_config

    @app.middleware("http")
    async def local_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "atlast",
            "version": app.version,
            "local_only": True,
            "ollama_model": web_config.ollama_model,
            "max_upload_bytes": web_config.max_upload_bytes,
            "max_context_chars": web_config.max_context_chars,
        }

    @app.post("/api/documents")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = sanitize_filename(file.filename or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            await file.close()
            raise _api_error(415, "unsupported_type", "Only PDF and UTF-8 TXT files are supported.")
        if file.size is not None and file.size > web_config.max_upload_bytes:
            await file.close()
            raise _api_error(
                413,
                "upload_too_large",
                f"Upload exceeds the {web_config.max_upload_bytes} byte limit.",
            )

        temp_path = web_config.upload_dir / f".upload-{uuid4().hex}.part"
        final_path: Path | None = None
        keep_final = False
        try:
            size = await _save_limited_upload(file, temp_path, web_config.max_upload_bytes)
            if size == 0:
                raise _api_error(422, "empty_document", "The uploaded file is empty.")

            async with app.state.operation_lock:
                final_path = _available_upload_path(web_config.upload_dir, filename)
                temp_path.replace(final_path)
                try:
                    result = await run_in_threadpool(app.state.index.ingest, final_path)
                except EmptyDocumentError as exc:
                    raise _api_error(422, "empty_document", str(exc)) from exc
                except DocumentReadError as exc:
                    raise _api_error(422, "invalid_document", str(exc)) from exc
                except AtlastError as exc:
                    raise _api_error(500, "index_error", str(exc)) from exc

            if result.status == "duplicate":
                final_path.unlink(missing_ok=True)
            else:
                keep_final = True
            payload = result.to_dict()
            if result.status == "duplicate":
                payload["source"] = None
            payload["filename"] = filename
            payload["size_bytes"] = size
            return payload
        except UploadTooLargeError as exc:
            raise _api_error(
                413,
                "upload_too_large",
                f"Upload exceeds the {web_config.max_upload_bytes} byte limit.",
            ) from exc
        finally:
            await file.close()
            temp_path.unlink(missing_ok=True)
            if final_path is not None and not keep_final:
                final_path.unlink(missing_ok=True)

    @app.get("/api/documents")
    async def list_documents() -> dict[str, Any]:
        async with app.state.operation_lock:
            documents = await run_in_threadpool(app.state.index.list_documents)
        return {
            "count": len(documents),
            "documents": [document.to_dict() for document in documents],
        }

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, Any]:
        try:
            async with app.state.operation_lock:
                deleted_chunks = await run_in_threadpool(
                    app.state.index.delete_document, document_id
                )
        except DocumentNotFoundError as exc:
            raise _api_error(404, "document_not_found", str(exc)) from exc
        except ValueError as exc:
            raise _api_error(422, "invalid_document_id", str(exc)) from exc
        except AtlastError as exc:
            raise _api_error(500, "index_error", str(exc)) from exc
        return {
            "status": "deleted",
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
        }

    @app.put("/api/documents/{document_id}")
    async def replace_document(document_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        filename = sanitize_filename(file.filename or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            await file.close()
            raise _api_error(415, "unsupported_type", "Only PDF and UTF-8 TXT files are supported.")
        if file.size is not None and file.size > web_config.max_upload_bytes:
            await file.close()
            raise _api_error(
                413,
                "upload_too_large",
                f"Upload exceeds the {web_config.max_upload_bytes} byte limit.",
            )

        temp_path = web_config.upload_dir / f".upload-{uuid4().hex}.part"
        final_path: Path | None = None
        keep_final = False
        try:
            size = await _save_limited_upload(file, temp_path, web_config.max_upload_bytes)
            if size == 0:
                raise _api_error(422, "empty_document", "The uploaded file is empty.")

            async with app.state.operation_lock:
                final_path = _available_upload_path(web_config.upload_dir, filename)
                temp_path.replace(final_path)
                try:
                    result = await run_in_threadpool(
                        app.state.index.replace_document, document_id, final_path
                    )
                except DocumentNotFoundError as exc:
                    raise _api_error(404, "document_not_found", str(exc)) from exc
                except DuplicateDocumentError as exc:
                    raise _api_error(409, "duplicate_replacement", str(exc)) from exc
                except EmptyDocumentError as exc:
                    raise _api_error(422, "empty_document", str(exc)) from exc
                except DocumentReadError as exc:
                    raise _api_error(422, "invalid_document", str(exc)) from exc
                except ValueError as exc:
                    raise _api_error(422, "invalid_document_id", str(exc)) from exc
                except AtlastError as exc:
                    raise _api_error(500, "index_error", str(exc)) from exc

            if result.status == "replaced":
                keep_final = True
            payload = result.to_dict()
            if result.status == "duplicate":
                payload["source"] = None
            payload["filename"] = filename
            payload["size_bytes"] = size
            return payload
        except UploadTooLargeError as exc:
            raise _api_error(
                413,
                "upload_too_large",
                f"Upload exceeds the {web_config.max_upload_bytes} byte limit.",
            ) from exc
        finally:
            await file.close()
            temp_path.unlink(missing_ok=True)
            if final_path is not None and not keep_final:
                final_path.unlink(missing_ok=True)

    @app.post("/api/ask")
    async def ask(payload: AskRequest):
        async with app.state.operation_lock:
            service = QuestionAnsweringService(
                app.state.index,
                app.state.generator,
                retrieval_limit=payload.limit,
                max_context_chars=web_config.max_context_chars,
            )
            response = await run_in_threadpool(service.ask, payload.question)
        response_payload = response.to_dict()
        response_payload["trace"]["model"] = web_config.ollama_model
        response_payload["trace"]["retrieval_limit"] = payload.limit
        response_payload["trace"]["max_context_chars"] = web_config.max_context_chars
        status_code = {
            "model_unavailable": 503,
            "generation_error": 502,
        }.get(response.status, 200)
        return JSONResponse(response_payload, status_code=status_code)

    @app.post("/api/retrieve")
    async def retrieve(payload: RetrievalRequest) -> dict[str, Any]:
        query = payload.query.strip()
        if not query:
            raise _api_error(422, "empty_query", "Retrieval query must not be empty.")
        async with app.state.operation_lock:
            hits = await run_in_threadpool(
                lambda: app.state.index.search(query, limit=payload.limit)
            )
        return {
            "query": query,
            "count": len(hits),
            "passages": [
                _retrieval_passage(hit, number)
                for number, hit in enumerate(hits, start=1)
            ],
        }

    static_dir = Path(__file__).with_name("static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="interface")
    return app


def sanitize_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    normalized = unicodedata.normalize("NFKC", basename)
    suffix = Path(normalized).suffix.lower()
    stem = Path(normalized).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-")
    if not safe_stem:
        safe_stem = "document"
    return f"{safe_stem[:96]}{suffix}"


def run_server(config: WebConfig, *, port: int = 8000) -> None:
    """Run the interface on loopback only."""

    import uvicorn

    uvicorn.run(create_app(config), host="127.0.0.1", port=port)


class UploadTooLargeError(Exception):
    pass


async def _save_limited_upload(file: UploadFile, path: Path, limit: int) -> int:
    size = 0
    with path.open("xb") as output:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                raise UploadTooLargeError
            output.write(chunk)
    return size


def _available_upload_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    source = Path(filename)
    return directory / f"{source.stem}-{uuid4().hex[:8]}{source.suffix}"


def _retrieval_passage(hit: SearchHit, number: int) -> dict[str, Any]:
    metadata = hit.metadata
    return {
        "passage_id": f"R{number}",
        "text": hit.text,
        "score": hit.score,
        "document_id": metadata.get("document_id"),
        "document_title": metadata.get("title")
        or metadata.get("filename")
        or "Unknown document",
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "section": metadata.get("section"),
        "chunk_index": metadata.get("chunk_index"),
    }


def _api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


app = create_app()
