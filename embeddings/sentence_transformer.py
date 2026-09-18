from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.config import settings

_MODEL_SINGLETON: Optional[SentenceTransformer] = None


def get_sentence_transformer_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model instance on CPU."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device="cpu",
        )
    return _MODEL_SINGLETON


class BGESentenceTransformerEmbedder:
    """Production embedder implementation using BAAI/bge-small-en-v1.5."""

    # BGE models use a specific query prefix for asymmetric search
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return settings.VECTOR_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Embed document chunk texts without query prefix.
        Returns float32 numpy array of shape (N, 384) with L2-normalized rows.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = get_sentence_transformer_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string with BGE instruction prefix.
        Returns float32 numpy array of shape (1, 384) with L2-normalized row.
        """
        prefixed_query = f"{self.BGE_QUERY_PREFIX}{query.strip()}"
        model = get_sentence_transformer_model()
        embedding = model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
