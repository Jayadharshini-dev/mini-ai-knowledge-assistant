from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.models import Citation, RetrievedChunk
from backend.rag.events import EventType, TraceEvent


class PipelineOutcome(BaseModel):
    """Final outcome summary of a pipeline run."""

    question: str
    answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    selected_chunks: list[RetrievedChunk] = Field(default_factory=list)
    rejected_chunks: list[RetrievedChunk] = Field(default_factory=list)
    abstained: bool = False
    degraded: Optional[str] = None
    skip_reason: Optional[str] = None
    elapsed_ms: int = 0
    timings: dict[str, int] = Field(default_factory=dict)


class CollectedRun(BaseModel):
    """Consumer-side helper container for a drained pipeline run."""

    events: list[TraceEvent]
    outcome: PipelineOutcome


async def collect_run(run_gen: AsyncIterator[TraceEvent]) -> CollectedRun:
    """
    Consumer-side helper function that drains an AsyncIterator[TraceEvent] run,
    collects all emitted events, and extracts the final PipelineOutcome.
    """
    events: list[TraceEvent] = []
    question = ""

    async for event in run_gen:
        events.append(event)
        if event.type == EventType.QUERY_RECEIVED:
            question = str(event.detail.get("question", ""))

    if not events:
        raise ValueError("Pipeline run emitted no events.")

    terminal_event = events[-1]
    if terminal_event.type not in (EventType.COMPLETE, EventType.ERROR):
        raise ValueError(
            f"Pipeline stream ended with non-terminal event '{terminal_event.type}'. "
            "Every run must end with COMPLETE or ERROR."
        )

    detail = terminal_event.detail

    if terminal_event.type == EventType.ERROR:
        outcome = PipelineOutcome(
            question=question,
            answer=None,
            citations=[],
            selected_chunks=[],
            rejected_chunks=[],
            abstained=False,
            degraded="error",
            skip_reason=str(detail.get("code", "ERROR")),
            elapsed_ms=terminal_event.t_ms,
            timings={},
        )
        return CollectedRun(events=events, outcome=outcome)

    # For COMPLETE event
    answer = detail.get("answer")
    abstained = bool(detail.get("abstained", False))
    degraded = detail.get("degraded")
    elapsed_ms = int(detail.get("elapsed_ms", terminal_event.t_ms))
    timings = detail.get("timings", {})

    raw_citations = detail.get("citations", [])
    citations = [Citation(**c) if isinstance(c, dict) else c for c in raw_citations]

    raw_chunks = detail.get("chunks", [])
    parsed_chunks = [
        RetrievedChunk(**c) if isinstance(c, dict) else c for c in raw_chunks
    ]

    selected_chunks = [c for c in parsed_chunks if c.above_threshold]
    rejected_chunks = [c for c in parsed_chunks if not c.above_threshold]

    outcome = PipelineOutcome(
        question=question,
        answer=answer,
        citations=citations,
        selected_chunks=selected_chunks,
        rejected_chunks=rejected_chunks,
        abstained=abstained,
        degraded=degraded,
        skip_reason=None if not abstained else "no_passage_above_threshold",
        elapsed_ms=elapsed_ms,
        timings=timings,
    )

    return CollectedRun(events=events, outcome=outcome)
