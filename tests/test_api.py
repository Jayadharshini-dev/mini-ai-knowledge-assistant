from __future__ import annotations

import io
import json
from collections.abc import Generator
from unittest.mock import MagicMock

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.deps import (
    get_document_registry,
    reset_dependencies,
    set_test_dependencies,
)
from backend.app.main import app
from backend.app.models import Chunk, DocumentRecord
from backend.generation.fake_provider import FakeLLMProvider
from kb.store import DocumentRegistry
from retrieval.vector_store import VectorStore


class ConstantVectorEmbedder:
    dimension = 384
    model_name = "constant-unit-embedder"

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones((1, 384), dtype=np.float32) / np.sqrt(384)


class DummyDocumentRegistry:
    def __init__(self, docs: list[DocumentRecord] | None = None):
        self.documents = {d.doc_id: d for d in (docs or [])}

    def load(self) -> None:
        pass

    def is_duplicate(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def register(
        self,
        doc_id: str,
        filename: str,
        pages: int,
        chunks: int,
        status: str = "indexed",
    ) -> DocumentRecord:
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
            chunks=chunks,
            indexed_at="2026-09-19T10:00:00Z",
            status=status,
        )
        self.documents[doc_id] = record
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self.documents.values(), key=lambda d: d.filename)


def _make_dummy_chunk(chunk_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"edge_doc__p001__c{chunk_index:04d}",
        doc_id="sha123",
        document="edge_doc.pdf",
        page=1,
        chunk_index=chunk_index,
        text=text,
        char_count=len(text),
        token_estimate=10,
    )


def _create_minimal_pdf_bytes(text: str = "Sample document content for test.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def clean_dependencies() -> Generator[None, None, None]:
    reset_dependencies()
    yield
    reset_dependencies()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_no_absolute_paths(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "config" in data
    config = data["config"]
    assert "GEMINI_API_KEY_SET" in config
    assert "INDEX_DIR" not in config
    assert "DOCUMENTS_DIR" not in config


def test_kb_status_empty(client: TestClient) -> None:
    empty_store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    empty_registry = DummyDocumentRegistry([])
    set_test_dependencies(
        store=empty_store,
        registry=empty_registry,
        provider=FakeLLMProvider(),
    )

    res = client.get("/api/kb/status")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "empty"
    assert data["documents"] == 0
    assert data["chunks"] == 0
    assert data["retriever"] == "dense"

    # Verify endpoint alias /api/knowledge-base/status
    res_alias = client.get("/api/knowledge-base/status")
    assert res_alias.status_code == 200
    assert res_alias.json() == data


def test_kb_status_ready(client: TestClient) -> None:
    store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()
    chunk = _make_dummy_chunk(1, "Test chunk content.")
    store.add_chunks([chunk], embedder.embed_chunks([chunk.text]))

    doc_record = DocumentRecord(
        doc_id="sha123",
        filename="edge_doc.pdf",
        pages=1,
        chunks=1,
        indexed_at="2026-09-19T10:00:00Z",
        status="indexed",
    )
    registry = DummyDocumentRegistry([doc_record])
    set_test_dependencies(
        store=store,
        registry=registry,
        embedder=embedder,
        provider=FakeLLMProvider(),
    )

    res = client.get("/api/kb/status")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "ready"
    assert data["documents"] == 1
    assert data["chunks"] == 1


def test_list_documents(client: TestClient) -> None:
    doc_record = DocumentRecord(
        doc_id="sha123",
        filename="edge_doc.pdf",
        pages=1,
        chunks=1,
        indexed_at="2026-09-19T10:00:00Z",
        status="indexed",
    )
    registry = DummyDocumentRegistry([doc_record])
    set_test_dependencies(registry=registry)

    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert len(data["documents"]) == 1
    assert data["documents"][0]["filename"] == "edge_doc.pdf"


def test_chat_validation_empty_question(client: TestClient) -> None:
    res = client.post("/api/chat", json={"question": "   "})
    assert res.status_code == 400
    data = res.json()
    assert data["code"] == "EMPTY_QUESTION"


def test_chat_validation_empty_kb(client: TestClient) -> None:
    empty_store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    set_test_dependencies(store=empty_store)

    res = client.post("/api/chat", json={"question": "What is edge computing?"})
    assert res.status_code == 409
    data = res.json()
    assert data["code"] == "NO_DOCUMENTS"


def test_chat_json_success(client: TestClient) -> None:
    store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()
    chunk = _make_dummy_chunk(1, "Edge computing reduces latency.")
    store.add_chunks([chunk], embedder.embed_chunks([chunk.text]))

    set_test_dependencies(
        store=store,
        embedder=embedder,
        provider=FakeLLMProvider(),
    )

    res = client.post(
        "/api/chat",
        json={
            "question": "What is edge computing?",
            "history": [{"role": "user", "content": "hello"}],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "trace" in data
    assert len(data["trace"]) > 0
    assert data["trace"][-1]["type"] == "COMPLETE"


def test_chat_stream_sse_ordering_and_framing(client: TestClient) -> None:
    """Verify real SSE stream, event ordering, framing, and headers."""
    store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()
    chunk = _make_dummy_chunk(1, "Edge computing reduces latency.")
    store.add_chunks([chunk], embedder.embed_chunks([chunk.text]))

    set_test_dependencies(
        store=store,
        embedder=embedder,
        provider=FakeLLMProvider(),
    )

    res = client.post(
        "/api/chat/stream",
        json={"question": "What is edge computing?"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["cache-control"] == "no-cache"

    # Parse SSE events from response text
    raw_text = res.text
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    assert len(blocks) > 0

    events = []
    for block in blocks:
        lines = block.split("\n")
        assert len(lines) >= 2
        event_line = lines[0]
        data_line = lines[1]

        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")

        event_name = event_line.replace("event: ", "").strip()
        data_json = json.loads(data_line.replace("data: ", "").strip())

        events.append((event_name, data_json))

    # Verify event names framing (trace vs done)
    for i, (evt_name, evt_data) in enumerate(events):
        if i < len(events) - 1:
            assert evt_name == "trace"
            assert evt_data["type"] not in ("COMPLETE", "ERROR")
        else:
            assert evt_name == "done"
            assert evt_data["type"] == "COMPLETE"

        # Verify no seq=999 or t_ms=0 fallback
        assert evt_data.get("seq") != 999

    # Verify sequence order
    types = [e[1]["type"] for e in events]
    expected_types = [
        "QUERY_RECEIVED",
        "RETRIEVAL_STARTED",
        "RETRIEVAL_COMPLETED",
        "EVIDENCE_SELECTED",
        "CONTEXT_BUILT",
        "GENERATION_STARTED",
        "GENERATION_COMPLETED",
        "COMPLETE",
    ]
    assert types == expected_types


def test_post_documents_real_ingestion_sse_stream(client: TestClient) -> None:
    """Verify POST /api/documents produces real ingestion SSE events."""
    store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()
    registry = DummyDocumentRegistry([])
    set_test_dependencies(store=store, registry=registry, embedder=embedder)

    pdf_bytes = _create_minimal_pdf_bytes("Edge computing reduces latency.")
    files = {"file": ("test_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    res = client.post("/api/documents", files=files)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    raw_text = res.text
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    events = []
    for block in blocks:
        lines = block.split("\n")
        event_name = lines[0].replace("event: ", "").strip()
        data_json = json.loads(lines[1].replace("data: ", "").strip())
        events.append((event_name, data_json))

    event_types = [e[1]["type"] for e in events]
    expected_types = [
        "DOCUMENT_RECEIVED",
        "TEXT_EXTRACTED",
        "TEXT_CLEANED",
        "CHUNKED",
        "EMBEDDED",
        "INDEXED",
        "COMPLETE",
    ]
    assert event_types == expected_types
    assert events[-1][0] == "done"
    assert events[-1][1]["type"] == "COMPLETE"


def test_post_documents_duplicate_detection(client: TestClient) -> None:
    """Verify POST /api/documents emits DOCUMENT_DUPLICATE for existing doc hash."""
    store = VectorStore(
        dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()

    pdf_bytes = _create_minimal_pdf_bytes("Duplicate test content.")
    from ingestion.chunker import _generate_doc_id

    doc_id = _generate_doc_id(pdf_bytes)

    existing_record = DocumentRecord(
        doc_id=doc_id,
        filename="existing.pdf",
        pages=1,
        chunks=1,
        indexed_at="2026-09-19T10:00:00Z",
        status="indexed",
    )
    registry = DummyDocumentRegistry([existing_record])
    set_test_dependencies(store=store, registry=registry, embedder=embedder)

    files = {"file": ("dup_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    res = client.post("/api/documents", files=files)
    assert res.status_code == 200

    raw_text = res.text
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    assert len(blocks) == 1

    lines = blocks[0].split("\n")
    assert lines[0] == "event: done"
    data = json.loads(lines[1].replace("data: ", ""))
    assert data["type"] == "DOCUMENT_DUPLICATE"


def test_registry_not_reloaded_on_every_request() -> None:
    """Verify B4: DocumentRegistry.load() is not called repeatedly on every request."""
    mock_reg = MagicMock(spec=DocumentRegistry)
    set_test_dependencies(registry=mock_reg)

    res1 = get_document_registry()
    res2 = get_document_registry()

    assert res1 is res2
    mock_reg.load.assert_not_called()


def test_error_masking_no_raw_exception_leakage(client: TestClient) -> None:
    """Verify that unhandled server exceptions do not leak raw exception details."""

    class BrokenRetriever:
        store = VectorStore(
            dimension=384, embedding_model="test-model", index_type="IndexFlatIP"
        )
        store.add_chunks(
            [_make_dummy_chunk(1, "c")],
            ConstantVectorEmbedder().embed_chunks(["c"]),
        )

        def retrieve(self, question: str) -> None:
            raise RuntimeError("SECRET_INTERNAL_DATABASE_PASSWORD_EXCEPTIONAL_FAILURE")

    from backend.rag.pipeline import RagPipeline

    broken_pipeline = RagPipeline(
        retriever=BrokenRetriever(),
        relevance=lambda x: None,
        context_builder=lambda **k: None,
        provider=FakeLLMProvider(),
        citation_validator=lambda **k: None,
    )

    set_test_dependencies(
        store=BrokenRetriever.store,
        provider=FakeLLMProvider(),
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.routes.get_pipeline", lambda: broken_pipeline)

        res = client.post("/api/chat", json={"question": "test?"})
        assert res.status_code == 500
        data = res.json()
        assert data["code"] in ("RETRIEVAL_FAILED", "INTERNAL_ERROR")
        assert "SECRET_INTERNAL_DATABASE_PASSWORD" not in json.dumps(data)
