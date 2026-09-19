from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import status
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.errors import AppException, ErrorCode
from backend.app.models import Chunk


class IndexManifest(BaseModel):
    embedding_model: str
    vector_dim: int
    index_type: str
    num_vectors: int
    chunk_count: int
    doc_ids: list[str] = Field(default_factory=list)
    created_at: str


class VectorStore:
    """
    FAISS IndexFlatIP vector store manager with atomic persistence,
    1:1 vector-to-metadata position alignment, and strict manifest mismatch guards.
    """

    def __init__(
        self,
        dimension: int | None = None,
        embedding_model: str | None = None,
        index_type: str | None = None,
    ):
        self.dimension = dimension or settings.VECTOR_DIM
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.index_type = index_type or settings.INDEX_TYPE

        # Initialize FAISS IndexFlatIP over inner product
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[Chunk] = []
        self.doc_ids: set[str] = set()

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their corresponding L2-normalized vectors to the FAISS index.
        Maintains 1:1 alignment: FAISS position i <-> chunks[i].
        """
        if not chunks:
            return

        if vectors.shape[0] != len(chunks):
            msg = f"Vector count ({vectors.shape[0]}) != chunk count ({len(chunks)})"
            raise ValueError(msg)

        if vectors.shape[1] != self.dimension:
            msg = f"Vector dimension {vectors.shape[1]} != store dim {self.dimension}"
            raise ValueError(msg)

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Add vectors to FAISS index
        self.index.add(vectors)
        self.chunks.extend(chunks)

        for chunk in chunks:
            self.doc_ids.add(chunk.doc_id)

    def save_index(self, index_dir: Path | str | None = None) -> Path:
        """Atomically persist FAISS index, metadata, and manifest."""
        target_dir = Path(index_dir or settings.INDEX_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = target_dir / "index.faiss"
        metadata_path = target_dir / "metadata.json"
        manifest_path = target_dir / "manifest.json"

        # 1. Build Manifest
        manifest = IndexManifest(
            embedding_model=self.embedding_model,
            vector_dim=self.dimension,
            index_type=self.index_type,
            num_vectors=self.num_vectors,
            chunk_count=len(self.chunks),
            doc_ids=sorted(self.doc_ids),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Atomic helper: write to temp file then rename
        def _atomic_write_bytes(dest_path: Path, data: bytes) -> None:
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(dest_path)

        def _atomic_write_json(dest_path: Path, data_dict: Any) -> None:
            json_bytes = json.dumps(data_dict, indent=2, ensure_ascii=False).encode(
                "utf-8"
            )
            _atomic_write_bytes(dest_path, json_bytes)

        # 2. Write FAISS Index
        tmp_faiss = faiss_path.with_suffix(".faiss.tmp")
        faiss.write_index(self.index, str(tmp_faiss))
        tmp_faiss.replace(faiss_path)

        # 3. Write Metadata JSON
        chunk_dicts = [c.model_dump() for c in self.chunks]
        _atomic_write_json(metadata_path, chunk_dicts)

        # 4. Write Manifest JSON
        _atomic_write_json(manifest_path, manifest.model_dump())

        return target_dir

    @classmethod
    def load_index(cls, index_dir: Path | str | None = None) -> VectorStore:
        """
        Load persisted FAISS index, metadata, and manifest with Mismatch Guard.
        """
        target_dir = Path(index_dir or settings.INDEX_DIR)
        faiss_path = target_dir / "index.faiss"
        metadata_path = target_dir / "metadata.json"
        manifest_path = target_dir / "manifest.json"

        all_exist = (
            faiss_path.exists() and metadata_path.exists() and manifest_path.exists()
        )
        if not all_exist:
            msg = f"Index files missing in '{target_dir}'."
            raise FileNotFoundError(msg)

        # 1. Load Manifest
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = IndexManifest(**manifest_data)
        except Exception as exc:
            msg = f"Failed to parse index manifest at '{manifest_path}': {exc}"
            raise AppException(
                code=ErrorCode.INDEX_MODEL_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        # 2. Manifest Mismatch Guard Checks
        if manifest.embedding_model != settings.EMBEDDING_MODEL:
            msg = (
                f"Index model '{manifest.embedding_model}' != "
                f"configured model '{settings.EMBEDDING_MODEL}'."
            )
            raise AppException(
                code=ErrorCode.INDEX_MODEL_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if manifest.vector_dim != settings.VECTOR_DIM:
            msg = (
                f"Index vector dim {manifest.vector_dim} != "
                f"configured dim {settings.VECTOR_DIM}."
            )
            raise AppException(
                code=ErrorCode.INDEX_DIMENSION_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if manifest.index_type != settings.INDEX_TYPE:
            msg = (
                f"Index type '{manifest.index_type}' != "
                f"configured type '{settings.INDEX_TYPE}'."
            )
            raise AppException(
                code=ErrorCode.INDEX_TYPE_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 3. Load FAISS Index
        try:
            faiss_index = faiss.read_index(str(faiss_path))
        except Exception as exc:
            msg = f"Failed to read FAISS index at '{faiss_path}': {exc}"
            raise AppException(
                code=ErrorCode.INDEX_MODEL_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        # 4. Load Chunk Metadata
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata_raw = json.load(f)
            chunks = [Chunk(**c) for c in metadata_raw]
        except Exception as exc:
            msg = f"Failed to read metadata at '{metadata_path}': {exc}"
            raise AppException(
                code=ErrorCode.INDEX_METADATA_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        # 5. Check Vector-to-Metadata Alignment Guard
        if faiss_index.ntotal != len(chunks) or manifest.num_vectors != len(chunks):
            msg = (
                f"Count mismatch: FAISS ntotal={faiss_index.ntotal}, "
                f"chunks={len(chunks)}, manifest={manifest.num_vectors}."
            )
            raise AppException(
                code=ErrorCode.INDEX_METADATA_MISMATCH,
                message=msg,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Reconstruct VectorStore instance
        store = cls(
            dimension=manifest.vector_dim,
            embedding_model=manifest.embedding_model,
            index_type=manifest.index_type,
        )
        store.index = faiss_index
        store.chunks = chunks
        store.doc_ids = set(manifest.doc_ids)

        return store
