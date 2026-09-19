from __future__ import annotations

import pytest

from backend.app.errors import AppException, ErrorCode
from backend.app.models import Chunk
from embeddings.fake import FakeEmbedder
from retrieval.retriever import DenseRetriever
from retrieval.vector_store import VectorStore


def _make_dummy_chunk(
    chunk_index: int, doc_id: str = "doc123", text: str = "Sample text"
) -> Chunk:
    return Chunk(
        chunk_id=f"test_doc__p001__c{chunk_index:04d}",
        doc_id=doc_id,
        document="test_doc.pdf",
        page=1,
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
        token_estimate=10,
    )


def test_dense_retriever_returns_top_k_ranked_results():
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)

    chunks = [
        _make_dummy_chunk(1, text="Edge computing architectures"),
        _make_dummy_chunk(2, text="Predictive maintenance machine learning"),
        _make_dummy_chunk(3, text="Campus resource management"),
    ]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    retriever = DenseRetriever(store=store, embedder=embedder)
    results = retriever.retrieve("What is edge computing?", top_k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert isinstance(results[0].score, float)
    assert results[0].chunk.document == "test_doc.pdf"


def test_dense_retriever_empty_question_raises_empty_question():
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    retriever = DenseRetriever(store=store, embedder=embedder)

    with pytest.raises(AppException) as exc_info:
        retriever.retrieve("   ")

    assert exc_info.value.code == ErrorCode.EMPTY_QUESTION
    assert exc_info.value.status_code == 400


def test_dense_retriever_empty_store_raises_no_documents():
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    retriever = DenseRetriever(store=store, embedder=embedder)

    with pytest.raises(AppException) as exc_info:
        retriever.retrieve("Valid question?")

    assert exc_info.value.code == ErrorCode.NO_DOCUMENTS
    assert exc_info.value.status_code == 409
