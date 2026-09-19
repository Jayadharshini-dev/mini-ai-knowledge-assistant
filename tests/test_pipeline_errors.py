from __future__ import annotations

import pytest

from backend.rag.events import EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from tests.conftest import assert_valid_event_stream


@pytest.mark.asyncio
async def test_pipeline_embedder_error_emits_safe_error_event(
    mock_pipeline_components,
):
    def failing_retrieve_embedding_step(question: str, top_k: int = 5):
        raise ValueError("Secret database password / internal stack trace")

    mock_pipeline_components["retriever"].retrieve = failing_retrieve_embedding_step

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Fail query"))
    events = collected.events

    assert_valid_event_stream(events)

    last_event = events[-1]
    assert last_event.type == EventType.ERROR

    # Requirement 5: Label contains only exception class name
    assert last_event.label == "Pipeline error (ValueError)"

    # Requirement 5: Detail contains safe user-facing message, not raw exception text
    err_detail = last_event.detail
    assert err_detail["code"] == "RETRIEVAL_FAILED"
    assert "Secret database password" not in err_detail["message"]
    assert err_detail["message"] == "An error occurred during vector retrieval."


@pytest.mark.asyncio
async def test_pipeline_retriever_error_emits_safe_error_event(
    mock_pipeline_components,
):
    def failing_retrieve(question: str, top_k: int = 5):
        raise RuntimeError("Raw FAISS pointer / memory address 0xdeadbeef")

    mock_pipeline_components["retriever"].retrieve = failing_retrieve

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Fail retrieve"))
    events = collected.events

    assert_valid_event_stream(events)

    last_event = events[-1]
    assert last_event.type == EventType.ERROR
    assert last_event.label == "Pipeline error (RuntimeError)"

    err_detail = last_event.detail
    assert err_detail["code"] == "RETRIEVAL_FAILED"
    assert "0xdeadbeef" not in err_detail["message"]
    assert err_detail["message"] == "An error occurred during vector retrieval."
