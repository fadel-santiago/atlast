"""Domain errors surfaced by the library and CLI."""


class AtlastError(Exception):
    """Base class for expected ATLAST failures."""


class DocumentReadError(AtlastError):
    """Raised when a source document cannot be read or parsed."""


class EmptyDocumentError(AtlastError):
    """Raised when a document has no indexable text."""


class IndexConfigurationError(AtlastError):
    """Raised when an existing index is incompatible with the embedder."""


class DocumentNotFoundError(AtlastError):
    """Raised when an indexed document ID does not exist."""


class DuplicateDocumentError(AtlastError):
    """Raised when replacement content is already indexed under another ID."""


class EvaluationError(AtlastError):
    """Raised when a retrieval evaluation dataset is invalid."""


class GenerationError(AtlastError):
    """Raised when a generation backend returns an unusable response."""


class OllamaUnavailableError(GenerationError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelUnavailableError(GenerationError):
    """Raised when the requested Ollama model is not installed."""
