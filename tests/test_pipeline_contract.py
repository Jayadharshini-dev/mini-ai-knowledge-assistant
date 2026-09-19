from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend.rag.events import EventEmitter, EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from tests.conftest import assert_valid_event_stream


def test_contracts_md_event_type_alignment():
    """Verify that all EventType enum members are documented in docs/CONTRACTS.md."""
    contracts_path = Path(__file__).parent.parent / "docs" / "CONTRACTS.md"
    assert contracts_path.exists(), "docs/CONTRACTS.md missing"

    content = contracts_path.read_text(encoding="utf-8")

    for e in EventType:
        assert e.value in content, (
            f"EventType member {e.value} not documented in CONTRACTS.md"
        )


@pytest.mark.asyncio
async def test_concurrent_independent_pipeline_runs(mock_pipeline_components):
    """
    Requirement 8: Both concurrent runs use the SAME RagPipeline instance to prove
    that EventEmitter is created inside run() for per-run isolation.
    """
    pipeline = RagPipeline(**mock_pipeline_components)

    c1, c2 = await asyncio.gather(
        collect_run(pipeline.run("Question 1")),
        collect_run(pipeline.run("Question 2")),
    )
    events1, events2 = c1.events, c2.events

    assert_valid_event_stream(events1)
    assert_valid_event_stream(events2)

    assert events1[0].seq == 1
    assert events2[0].seq == 1
    assert [e.seq for e in events1] == list(range(1, len(events1) + 1))
    assert [e.seq for e in events2] == list(range(1, len(events2) + 1))


@pytest.mark.asyncio
async def test_collect_run_rejects_non_terminal_final_event(
    mock_pipeline_components,
):
    """Requirement 10: collect_run rejects stream with non-terminal final event."""

    async def non_terminal_stream():
        emitter = EventEmitter()
        yield emitter.emit(
            EventType.QUERY_RECEIVED, "ok", "Received", {"question": "Test"}
        )

    with pytest.raises(ValueError, match="Every run must end with COMPLETE or ERROR"):
        await collect_run(non_terminal_stream())


@pytest.mark.asyncio
async def test_pipeline_generator_aclose_cancellation(mock_pipeline_components):
    """Verify early closing of pipeline async generator via aclose()."""
    pipeline = RagPipeline(**mock_pipeline_components)
    gen = pipeline.run("Cancellation question")

    first_event = await gen.__anext__()
    assert first_event.type == EventType.QUERY_RECEIVED

    await gen.aclose()

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_pipeline_reproducibility(mock_pipeline_components):
    """Verify identical inputs produce identical event sequences."""
    pipeline = RagPipeline(**mock_pipeline_components)

    c1 = await collect_run(pipeline.run("Deterministic question"))
    c2 = await collect_run(pipeline.run("Deterministic question"))

    types1 = [e.type for e in c1.events]
    types2 = [e.type for e in c2.events]
    assert types1 == types2
    assert c1.outcome.answer == c2.outcome.answer
