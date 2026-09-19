from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from backend.app.models import Chunk
from backend.generation.citation_validator import validate_citations
from backend.generation.context_builder import build_generation_context
from backend.generation.fake_provider import FakeLLMProvider
from backend.rag.events import EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from retrieval.relevance import evaluate_relevance
from retrieval.retriever import DenseRetriever
from retrieval.vector_store import VectorStore
from tests.conftest import assert_valid_event_stream


class ConstantVectorEmbedder:
    dimension = 384
    model_name = "constant-unit-embedder"

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        vec = np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)
        return vec

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.ones((1, 384), dtype=np.float32) / np.sqrt(384)
        return vec


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


@pytest.mark.asyncio
async def test_real_pipeline_production_collaborators_integration():
    """
    Requirement 5: Real Phase 05 integration test constructing actual
    VectorStore, DenseRetriever, evaluate_relevance partial binding,
    build_generation_context, validate_citations, and FakeLLMProvider.
    Tests production collaborator interfaces.
    """
    # 1. Setup real production collaborators with unit vectors (score = 1.0)
    store = VectorStore(
        dimension=384, embedding_model="constant-unit", index_type="IndexFlatIP"
    )
    embedder = ConstantVectorEmbedder()

    chunk1 = _make_dummy_chunk(
        1, "Edge computing reduces latency for IoT control loops."
    )
    chunk2 = _make_dummy_chunk(
        2, "Predictive maintenance algorithms analyze sensor data."
    )

    vectors = embedder.embed_chunks([chunk1.text, chunk2.text])
    store.add_chunks([chunk1, chunk2], vectors)

    retriever = DenseRetriever(store=store, embedder=embedder)
    relevance_fn = partial(evaluate_relevance, threshold=0.67)

    pipeline = RagPipeline(
        retriever=retriever,
        relevance=relevance_fn,
        context_builder=build_generation_context,
        provider=FakeLLMProvider(),
        citation_validator=validate_citations,
    )

    # 2. Run query stream
    collected = await collect_run(pipeline.run("What is edge computing latency?"))
    events, outcome = collected.events, collected.outcome

    # 3. Assert stream validity and exact sequence
    assert_valid_event_stream(events)

    event_types = [e.type for e in events]
    expected_types = [
        EventType.QUERY_RECEIVED,
        EventType.RETRIEVAL_STARTED,
        EventType.RETRIEVAL_COMPLETED,
        EventType.EVIDENCE_SELECTED,
        EventType.CONTEXT_BUILT,
        EventType.GENERATION_STARTED,
        EventType.GENERATION_COMPLETED,
        EventType.COMPLETE,
    ]
    assert event_types == expected_types

    # 4. Assert outcome and interface properties
    assert outcome.abstained is False
    assert outcome.degraded is None
    assert outcome.answer is not None
    assert "edge computing" in outcome.answer.lower()
    assert len(outcome.selected_chunks) == 2
    assert len(outcome.citations) > 0

    # Verify QUERY_RECEIVED and RETRIEVAL_COMPLETED used real VectorStore properties
    assert events[0].detail["index_size"] == 2
    assert events[2].detail["index_type"] == "IndexFlatIP"
    assert events[3].detail["threshold"] == 0.67
