from __future__ import annotations

import logging
from functools import partial
from typing import Any

from backend.app.config import settings
from backend.app.models import KnowledgeBaseStatus
from backend.generation.citation_validator import validate_citations
from backend.generation.context_builder import build_generation_context
from backend.generation.fake_provider import FakeLLMProvider
from backend.generation.gemini_provider import GeminiLLMProvider
from backend.rag.pipeline import RagPipeline
from embeddings.sentence_transformer import BGESentenceTransformerEmbedder
from kb.store import DocumentRegistry
from retrieval.relevance import evaluate_relevance
from retrieval.retriever import DenseRetriever
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_store: VectorStore | None = None
_registry: DocumentRegistry | None = None
_embedder: Any | None = None
_provider: Any | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        index_dir = settings.INDEX_DIR
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            try:
                _store = VectorStore.load(index_dir)
            except Exception:
                logger.exception("Failed to load VectorStore from %s", index_dir)
                _store = VectorStore(
                    dimension=settings.VECTOR_DIM,
                    embedding_model=settings.EMBEDDING_MODEL,
                    index_type=settings.INDEX_TYPE,
                )
        else:
            _store = VectorStore(
                dimension=settings.VECTOR_DIM,
                embedding_model=settings.EMBEDDING_MODEL,
                index_type=settings.INDEX_TYPE,
            )
    return _store


def get_document_registry() -> DocumentRegistry:
    global _registry
    if _registry is None:
        _registry = DocumentRegistry(settings.INDEX_DIR)
    return _registry


def get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        _embedder = BGESentenceTransformerEmbedder(model_name=settings.EMBEDDING_MODEL)
    return _embedder


def get_provider() -> Any:
    global _provider
    if _provider is None:
        if settings.LLM_PROVIDER.lower() == "gemini":
            _provider = GeminiLLMProvider()
        else:
            _provider = FakeLLMProvider()
    return _provider


def set_test_dependencies(
    store: VectorStore | None = None,
    registry: DocumentRegistry | None = None,
    embedder: Any | None = None,
    provider: Any | None = None,
) -> None:
    """Helper to inject test doubles for API tests."""
    global _store, _registry, _embedder, _provider
    _store = store
    _registry = registry
    _embedder = embedder
    _provider = provider


def reset_dependencies() -> None:
    """Reset dependency singletons."""
    global _store, _registry, _embedder, _provider
    _store = None
    _registry = None
    _embedder = None
    _provider = None


def get_pipeline(
    store: VectorStore | None = None,
    embedder: Any | None = None,
    provider: Any | None = None,
) -> RagPipeline:
    v_store = store or get_vector_store()
    v_embedder = embedder or get_embedder()
    v_provider = provider or get_provider()

    retriever = DenseRetriever(store=v_store, embedder=v_embedder)
    relevance_fn = partial(evaluate_relevance, threshold=settings.RELEVANCE_THRESHOLD)

    return RagPipeline(
        retriever=retriever,
        relevance=relevance_fn,
        context_builder=build_generation_context,
        provider=v_provider,
        citation_validator=validate_citations,
    )


def get_kb_status(
    store: VectorStore | None = None,
    registry: DocumentRegistry | None = None,
    provider: Any | None = None,
) -> KnowledgeBaseStatus:
    v_store = store or get_vector_store()
    v_registry = registry or get_document_registry()
    v_provider = provider or get_provider()

    docs_count = len(v_registry.documents)
    chunks_count = v_store.num_vectors

    if docs_count == 0 or chunks_count == 0:
        state = "empty"
    else:
        state = "ready"

    index_file = settings.INDEX_DIR / "faiss.index"
    index_size_bytes = index_file.stat().st_size if index_file.is_file() else 0

    return KnowledgeBaseStatus(
        state=state,
        documents=docs_count,
        chunks=chunks_count,
        vector_dim=settings.VECTOR_DIM,
        index_type=v_store.index_type,
        index_size_bytes=index_size_bytes,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        llm_available=v_provider.is_available(),
        top_k=settings.TOP_K,
        relevance_threshold=settings.RELEVANCE_THRESHOLD,
        retriever="dense",
    )
