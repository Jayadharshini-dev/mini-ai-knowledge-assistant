from __future__ import annotations

import json

import pytest

from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline


@pytest.mark.asyncio
async def test_real_pipeline_complete_payload_contract(mock_pipeline_components):
    """Verify real RagPipeline COMPLETE event matches frontend expectations."""
    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("What is edge computing?"))

    events = collected.events
    complete_event = events[-1]

    assert complete_event.type.value == "COMPLETE"
    detail = complete_event.detail

    assert "answer" in detail
    assert "citations" in detail
    assert "chunks" in detail
    assert "elapsed_ms" in detail
    assert "abstained" in detail
    assert "degraded" in detail
    assert isinstance(detail["abstained"], bool)
    assert detail["abstained"] is False
    assert detail["degraded"] is None

    timings = detail["timings"]
    assert "search_ms" in timings
    assert "context_ms" in timings
    assert "generate_ms" in timings
    assert "embed_ms" not in timings


@pytest.mark.asyncio
async def test_real_pipeline_abstention_payload_contract(mock_pipeline_components):
    """Verify real RagPipeline ABSTAINED event and COMPLETE detail for low relevance."""
    from retrieval.relevance import RelevanceDecision

    mock_pipeline_components["relevance"] = lambda chunks: RelevanceDecision(
        decision="abstain",
        top_score=0.1,
        threshold=0.67,
        selected_chunks=[],
        rejected_chunks=chunks,
        reason="no_passage_above_threshold",
        message="No relevant passage found.",
    )

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Irrelevant question"))

    events = collected.events
    types = [e.type.value for e in events]

    assert "ABSTAINED" in types
    assert types[-1] == "COMPLETE"

    abstained_event = [e for e in events if e.type.value == "ABSTAINED"][0]
    assert abstained_event.detail["reason"] == "no_passage_above_threshold"
    assert abstained_event.detail["threshold"] == 0.67

    complete_detail = events[-1].detail
    assert complete_detail["abstained"] is True
    assert complete_detail["answer"] is None


@pytest.mark.asyncio
async def test_real_pipeline_degraded_no_provider_payload_contract(
    mock_pipeline_components,
):
    """Verify GENERATION_SKIPPED and COMPLETE detail for missing provider."""

    class UnavailableProvider:
        provider_name = "fake"
        model_name = "fake-model"

        def is_available(self) -> bool:
            return False

    mock_pipeline_components["provider"] = UnavailableProvider()

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Question for degraded provider"))

    events = collected.events
    types = [e.type.value for e in events]

    assert "GENERATION_SKIPPED" in types
    assert types[-1] == "COMPLETE"

    complete_detail = events[-1].detail
    assert complete_detail["abstained"] is False
    assert complete_detail["degraded"] == "no_provider"
    assert complete_detail["answer"] is None


def test_sse_chunked_network_stream_parsing():
    """Verify SSE chunk splitting and buffer reassembly contract."""
    f1 = (
        'event: trace\ndata: {"seq":1,"type":"QUERY_RECEIVED",'
        '"status":"ok","t_ms":1,"label":"Received","detail":{}}\n\n'
    )
    f2 = (
        'event: done\ndata: {"seq":2,"type":"COMPLETE",'
        '"status":"ok","t_ms":10,"label":"Complete",'
        '"detail":{"answer":"ok","citations":[],"chunks":[],'
        '"elapsed_ms":10,"abstained":false,"degraded":null}}\n\n'
    )

    # Stream split across TCP chunks
    chunk1 = f1 + f2[:20]
    chunk2 = f2[20:]

    buffer = chunk1
    blocks = buffer.split("\n\n")
    buffer = blocks.pop()

    assert len(blocks) == 1
    d1 = json.loads(blocks[0].split("\n")[1].replace("data: ", ""))
    assert d1["type"] == "QUERY_RECEIVED"

    buffer += chunk2
    blocks = buffer.split("\n\n")
    buffer = blocks.pop()

    assert len(blocks) == 1
    d2 = json.loads(blocks[0].split("\n")[1].replace("data: ", ""))
    assert d2["type"] == "COMPLETE"
