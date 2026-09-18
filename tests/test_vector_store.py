from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.config import settings
from backend.app.errors import AppException, ErrorCode
from backend.app.models import Chunk
from embeddings.fake import FakeEmbedder
from kb.store import DocumentRegistry
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


def test_vector_store_add_chunks_and_alignment():
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)

    chunks = [_make_dummy_chunk(1), _make_dummy_chunk(2)]
    texts = [c.text for c in chunks]
    vectors = embedder.embed_chunks(texts)

    store.add_chunks(chunks, vectors)

    assert store.num_vectors == 2
    assert len(store.chunks) == 2
    assert store.chunks[0].chunk_id == chunks[0].chunk_id
    assert store.chunks[1].chunk_id == chunks[1].chunk_id


def test_vector_store_rejects_mismatched_vector_dimension():
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=128)  # Incorrect dimension

    chunks = [_make_dummy_chunk(1)]
    vectors = embedder.embed_chunks([c.text for c in chunks])

    msg_pat = "Vector dimension 128 != store dim 384"
    with pytest.raises(ValueError, match=msg_pat):
        store.add_chunks(chunks, vectors)


def test_vector_store_atomic_save_and_reload_roundtrip(tmp_path: Path):
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)

    chunks = [_make_dummy_chunk(1), _make_dummy_chunk(2)]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)

    # Save to temporary index directory
    save_dir = store.save_index(tmp_path)

    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "manifest.json").exists()

    # Reload store
    reloaded_store = VectorStore.load_index(save_dir)

    assert reloaded_store.num_vectors == 2
    assert len(reloaded_store.chunks) == 2
    assert reloaded_store.chunks[0].chunk_id == chunks[0].chunk_id
    assert reloaded_store.dimension == 384
    assert reloaded_store.embedding_model == settings.EMBEDDING_MODEL


def test_manifest_mismatch_guard_detects_incompatible_model(tmp_path: Path):
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    chunks = [_make_dummy_chunk(1)]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)
    save_dir = store.save_index(tmp_path)

    # Tamper with manifest to simulate incompatible model name
    manifest_path = save_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["embedding_model"] = "INCOMPATIBLE/different-model-v2"
    manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(AppException) as exc_info:
        VectorStore.load_index(save_dir)

    assert exc_info.value.code == ErrorCode.INDEX_MODEL_MISMATCH
    assert exc_info.value.status_code == 500
    assert "Index model" in exc_info.value.message


def test_manifest_mismatch_guard_detects_dimension_mismatch(tmp_path: Path):
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    chunks = [_make_dummy_chunk(1)]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)
    save_dir = store.save_index(tmp_path)

    # Tamper with manifest to simulate dimension mismatch
    manifest_path = save_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["vector_dim"] = 768
    manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(AppException) as exc_info:
        VectorStore.load_index(save_dir)

    assert exc_info.value.code == ErrorCode.INDEX_DIMENSION_MISMATCH
    assert exc_info.value.status_code == 500
    assert "Index vector dim" in exc_info.value.message


def test_manifest_mismatch_guard_detects_index_type_mismatch(tmp_path: Path):
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    chunks = [_make_dummy_chunk(1)]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)
    save_dir = store.save_index(tmp_path)

    # Tamper with manifest to simulate index type mismatch
    manifest_path = save_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["index_type"] = "IndexFlatL2"
    manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(AppException) as exc_info:
        VectorStore.load_index(save_dir)

    assert exc_info.value.code == ErrorCode.INDEX_TYPE_MISMATCH
    assert exc_info.value.status_code == 500
    assert "Index type" in exc_info.value.message


def test_manifest_mismatch_guard_detects_count_mismatch(tmp_path: Path):
    store = VectorStore(dimension=384)
    embedder = FakeEmbedder(dimension=384)
    chunks = [_make_dummy_chunk(1), _make_dummy_chunk(2)]
    vectors = embedder.embed_chunks([c.text for c in chunks])
    store.add_chunks(chunks, vectors)
    save_dir = store.save_index(tmp_path)

    # Tamper with metadata to remove one chunk
    metadata_path = save_dir / "metadata.json"
    metadata_data = json.loads(metadata_path.read_text())
    metadata_data.pop()  # Remove last entry
    metadata_path.write_text(json.dumps(metadata_data))

    with pytest.raises(AppException) as exc_info:
        VectorStore.load_index(save_dir)

    assert exc_info.value.code == ErrorCode.INDEX_METADATA_MISMATCH
    assert exc_info.value.status_code == 500
    assert "Count mismatch" in exc_info.value.message


def test_document_registry_deduplication(tmp_path: Path):
    registry = DocumentRegistry(registry_dir=tmp_path)
    doc_id = "doc_sha256_hash_123"

    assert not registry.is_duplicate(doc_id)

    registry.register(
        doc_id=doc_id,
        filename="edge_computing.pdf",
        pages=5,
        chunks=10,
    )

    assert registry.is_duplicate(doc_id)

    # Re-instantiate registry from same directory to test persistence
    reloaded_registry = DocumentRegistry(registry_dir=tmp_path)
    assert reloaded_registry.is_duplicate(doc_id)
    assert len(reloaded_registry.list_documents()) == 1
    assert reloaded_registry.list_documents()[0].filename == "edge_computing.pdf"
