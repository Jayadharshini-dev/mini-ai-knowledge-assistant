from __future__ import annotations

import hashlib

import numpy as np

from backend.app.config import settings


class FakeEmbedder:
    """
    Deterministic fake embedder for fast, offline unit testing.
    Generates reproducible 384-d L2-normalized float32 vectors.
    """

    def __init__(
        self,
        dimension: int | None = None,
        model_name: str | None = None,
    ):
        self._dimension = dimension or settings.VECTOR_DIM
        self._model_name = model_name or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_text_to_vector(self, text: str) -> np.ndarray:
        """Generate a deterministic L2-normalized vector from text SHA-256 seed."""
        sha = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed random generator with 4 bytes from hash
        seed = int.from_bytes(sha[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = [self._hash_text_to_vector(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._hash_text_to_vector(f"query:{query}")
        return vec.reshape(1, self.dimension).astype(np.float32)
