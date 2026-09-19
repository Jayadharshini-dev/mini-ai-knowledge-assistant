from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.models import Chunk, RetrievedChunk
from backend.rag.events import EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from tests.conftest import assert_valid_event_stream


@pytest.fixture
def low_score_chunk() -> RetrievedChunk:
    c = Chunk(
        chunk_id="edge_computing_iot__p001__c0001",
        doc_id="sha123",
        document="edge_computing_iot.pdf",
        page=1,
        chunk_index=1,
        text="Unrelated topic.",
        char_count=16,
        token_estimate=3,
    )
    return RetrievedChunk(chunk=c, score=0.20, rank=1, above_threshold=False)


@pytest.mark.asyncio
async def test_pipeline_abstention_flow_and_events(
    mock_pipeline_components, low_score_chunk: RetrievedChunk
):
    mock_pipeline_components["retriever"].retrieve = lambda q, top_k=5: [
        low_score_chunk
    ]

    provider_mock = MagicMock()
    mock_pipeline_components["provider"] = provider_mock

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("What is quantum gravity?"))
    events, outcome = collected.events, collected.outcome

    # 1. Event stream validity
    assert_valid_event_stream(events)

    # 2. Strict Abstention Invariant: provider generate never called
    assert provider_mock.generate.call_count == 0

    # 3. Exact event sequence
    event_types = [e.type for e in events]
    expected_types = [
        EventType.QUERY_RECEIVED,
        EventType.RETRIEVAL_STARTED,
        EventType.RETRIEVAL_COMPLETED,
        EventType.EVIDENCE_SELECTED,
        EventType.ABSTAINED,
        EventType.COMPLETE,
    ]
    assert event_types == expected_types

    # 4. Pipeline outcome check
    assert outcome.abstained is True
    assert outcome.answer is None
    assert outcome.citations == []
    assert outcome.selected_chunks == []
    assert len(outcome.rejected_chunks) == 1

    # Requirement 7: COMPLETE detail contains all retrieved chunks
    complete_event = events[-1]
    assert "chunks" in complete_event.detail
    assert len(complete_event.detail["chunks"]) == 1
    assert complete_event.detail["chunks"][0]["above_threshold"] is False


@pytest.mark.asyncio
async def test_pipeline_abstention_empty_retrieval(mock_pipeline_components):
    mock_pipeline_components["retriever"].retrieve = lambda q, top_k=5: []

    provider_mock = MagicMock()
    mock_pipeline_components["provider"] = provider_mock

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Empty retrieval question"))
    events, outcome = collected.events, collected.outcome

    assert_valid_event_stream(events)
    assert provider_mock.generate.call_count == 0
    assert outcome.abstained is True
    assert outcome.answer is None
