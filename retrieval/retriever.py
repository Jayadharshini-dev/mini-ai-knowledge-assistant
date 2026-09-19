from __future__ import annotations

import numpy as np
from fastapi import status

from backend.app.config import settings
from backend.app.errors import AppException, ErrorCode
from backend.app.models import RetrievedChunk
from embeddings.base import Embedder
from retrieval.vector_store import VectorStore


class DenseRetriever:
    """
    Dense vector retriever using BGE query embeddings and FAISS IndexFlatIP.
    Returns Top-K RetrievedChunk objects with cosine similarity scores.
    """

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            raise AppException(
                code=ErrorCode.EMPTY_QUESTION,
                message="Question cannot be empty or whitespace.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if self.store.num_vectors == 0:
            raise AppException(
                code=ErrorCode.NO_DOCUMENTS,
                message="No documents indexed in the knowledge base.",
                status_code=status.HTTP_409_CONFLICT,
            )

        k_val = top_k or settings.TOP_K
        k_val = min(k_val, self.store.num_vectors)
        thresh_val = (
            threshold if threshold is not None else settings.RELEVANCE_THRESHOLD
        )

        # Embed query with BGE prefix
        query_vec = self.embedder.embed_query(cleaned_question)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        # FAISS search
        scores, indices = self.store.index.search(query_vec, k_val)

        retrieved_chunks: list[RetrievedChunk] = []
        raw_scores = scores[0]
        raw_indices = indices[0]

        for idx, (faiss_pos, score_val) in enumerate(zip(raw_indices, raw_scores)):
            if faiss_pos < 0 or faiss_pos >= len(self.store.chunks):
                continue

            chunk_obj = self.store.chunks[faiss_pos]
            float_score = float(score_val)
            rank = idx + 1  # 1-based rank

            retrieved = RetrievedChunk(
                chunk=chunk_obj,
                score=float_score,
                rank=rank,
                above_threshold=bool(float_score >= thresh_val),
            )
            retrieved_chunks.append(retrieved)

        return retrieved_chunks
