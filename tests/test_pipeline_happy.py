from __future__ import annotations

import threading

import pytest

from backend.rag.events import EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from tests.conftest import assert_valid_event_stream


@pytest.mark.asyncio
async def test_pipeline_happy_path_event_stream(mock_pipeline_components):
    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("What is IoT latency?"))
    events, outcome = collected.events, collected.outcome

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

    assert outcome.abstained is False
    assert outcome.degraded is None
    assert outcome.answer is not None
    assert len(outcome.answer) > 0
    assert len(outcome.selected_chunks) > 0
    assert outcome.citations is not None

    # COMPLETE detail contains all retrieved chunks with preserved flags
    complete_event = events[-1]
    assert "chunks" in complete_event.detail
    assert len(complete_event.detail["chunks"]) == 1
    assert complete_event.detail["chunks"][0]["above_threshold"] is True


@pytest.mark.asyncio
async def test_pipeline_blocking_calls_run_in_worker_threads(mock_pipeline_components):
    main_thread_id = threading.get_ident()
    thread_ids_used = []

    original_retrieve = mock_pipeline_components["retriever"].retrieve
    original_generate = mock_pipeline_components["provider"].generate

    def spy_retrieve(question: str, top_k: int = 5):
        thread_ids_used.append(("retrieve", threading.get_ident()))
        return original_retrieve(question, top_k=top_k)

    def spy_generate(context):
        thread_ids_used.append(("generate", threading.get_ident()))
        return original_generate(context)

    mock_pipeline_components["retriever"].retrieve = spy_retrieve
    mock_pipeline_components["provider"].generate = spy_generate

    pipeline = RagPipeline(**mock_pipeline_components)
    _ = await collect_run(pipeline.run("Thread check question"))

    # retrieve & generate are offloaded via asyncio.to_thread
    assert len(thread_ids_used) == 2
    for operation, tid in thread_ids_used:
        assert tid != main_thread_id, (
            f"Operation {operation} ran on main event loop thread {main_thread_id}"
        )
