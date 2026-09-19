from __future__ import annotations

import json
from functools import partial

import pytest

from backend.app.models import Chunk, Citation, RetrievedChunk
from backend.generation.base import GenerationContext
from backend.generation.citation_validator import CitationValidationResult
from backend.generation.fake_provider import FakeLLMProvider
from backend.rag.events import EventType, TraceEvent
from retrieval.relevance import evaluate_relevance


def assert_valid_event_stream(events: list[TraceEvent]) -> None:
    """
    Validation helper asserting invariant stream properties:
    - seq == 1..N
    - t_ms non-decreasing
    - exactly ONE terminal event (COMPLETE or ERROR)
    - terminal event is the last event
    - every event passes json.dumps(event.to_dict())
    - event types belong to EventType enum
    """
    assert len(events) > 0, "Event stream must not be empty"

    # 1. Monotonic seq 1..N
    expected_seqs = list(range(1, len(events) + 1))
    actual_seqs = [e.seq for e in events]
    assert actual_seqs == expected_seqs, (
        f"Sequence mismatch: expected {expected_seqs}, got {actual_seqs}"
    )

    # 2. Non-decreasing t_ms
    for i in range(1, len(events)):
        prev_t = events[i - 1].t_ms
        curr_t = events[i].t_ms
        assert curr_t >= prev_t, (
            f"t_ms decreased at seq {events[i].seq}: {prev_t} > {curr_t}"
        )

    # 3. Terminal event invariants
    terminal_events = [
        e for e in events if e.type in (EventType.COMPLETE, EventType.ERROR)
    ]
    types_found = [e.type for e in terminal_events]
    assert len(terminal_events) == 1, (
        f"Expected 1 terminal event, found {len(terminal_events)}: {types_found}"
    )
    assert events[-1].type in (EventType.COMPLETE, EventType.ERROR), (
        f"Last event must be terminal (COMPLETE or ERROR), got {events[-1].type}"
    )

    # 4. Pure JSON serializability
    for event in events:
        try:
            dumped = json.dumps(event.to_dict())
            assert dumped is not None
        except Exception as exc:
            pytest.fail(
                f"Event seq {event.seq} ({event.type}) failed json.dumps: {exc}"
            )

    # 5. Non-terminal events check
    for event in events[:-1]:
        assert event.type not in (EventType.COMPLETE, EventType.ERROR), (
            f"Terminal event {event.type} emitted before end at seq {event.seq}"
        )


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    c = Chunk(
        chunk_id="edge_computing_iot__p001__c0001",
        doc_id="sha123",
        document="edge_computing_iot.pdf",
        page=1,
        chunk_index=1,
        text="Latency matters for control loops.",
        char_count=34,
        token_estimate=7,
    )
    return RetrievedChunk(chunk=c, score=0.85, rank=1, above_threshold=True)


@pytest.fixture
def mock_pipeline_components(sample_chunk: RetrievedChunk):
    class MockStore:
        index_type = "IndexFlatIP"
        num_vectors = 42

    class MockRetriever:
        store = MockStore()

        def retrieve(self, question: str, top_k: int = 5):
            return [sample_chunk]

    mock_relevance = partial(evaluate_relevance, threshold=0.67)

    def mock_context_builder(
        question: str, chunks: list[RetrievedChunk]
    ) -> GenerationContext:
        return GenerationContext(
            question=question,
            selected_chunks=chunks,
            formatted_prompt="Grounded prompt",
            passages_count=len(chunks),
            context_chars=sum(len(c.chunk.text) for c in chunks),
            token_estimate=10,
            truncated=False,
        )

    def mock_citation_validator(
        raw_text: str, chunks: list[RetrievedChunk]
    ) -> CitationValidationResult:
        cites = (
            [
                Citation(
                    n=1,
                    chunk_id=chunks[0].chunk.chunk_id,
                    document=chunks[0].chunk.document,
                    page=chunks[0].chunk.page,
                )
            ]
            if chunks
            else []
        )
        return CitationValidationResult(
            clean_text=raw_text,
            valid_citations=cites,
            citations_emitted=len(cites),
            citations_dropped=0,
        )

    return {
        "retriever": MockRetriever(),
        "relevance": mock_relevance,
        "context_builder": mock_context_builder,
        "provider": FakeLLMProvider(),
        "citation_validator": mock_citation_validator,
    }
