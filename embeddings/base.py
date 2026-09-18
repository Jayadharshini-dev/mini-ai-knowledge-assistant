from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol defining the interface for document and query embedding generators."""

    @property
    def dimension(self) -> int:
        """Vector dimension (e.g. 384 for bge-small-en-v1.5)."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        ...

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Embed document chunk texts.
        Returns a float32 numpy array of shape (N, dimension) with L2-normalized rows.
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query.
        Returns a float32 numpy array of shape (1, dimension) with L2-normalized row.
        """
        ...
