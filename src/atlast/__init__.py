"""ATLAST local document indexing foundation."""

from atlast.errors import AtlastError, DocumentReadError, EmptyDocumentError
from atlast.index import IngestResult, SearchHit, VectorIndex
from atlast.qa import AnswerResponse, Citation, OllamaGenerator, QuestionAnsweringService

__all__ = [
    "AtlastError",
    "AnswerResponse",
    "Citation",
    "DocumentReadError",
    "EmptyDocumentError",
    "IngestResult",
    "OllamaGenerator",
    "QuestionAnsweringService",
    "SearchHit",
    "VectorIndex",
]
