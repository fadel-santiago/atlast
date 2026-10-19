"""Embedding interfaces and the production Sentence Transformers adapter."""

from __future__ import annotations

from typing import Protocol, Sequence


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """CPU-friendly local embeddings, loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        dimension = self._load_model().get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Embedding model did not report a dimension: {self.name}")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model
