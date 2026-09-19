from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Query stream
    QUERY_RECEIVED = "QUERY_RECEIVED"
    RETRIEVAL_STARTED = "RETRIEVAL_STARTED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    EVIDENCE_SELECTED = "EVIDENCE_SELECTED"
    ABSTAINED = "ABSTAINED"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    GENERATION_STARTED = "GENERATION_STARTED"
    GENERATION_COMPLETED = "GENERATION_COMPLETED"
    GENERATION_SKIPPED = "GENERATION_SKIPPED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

    # Ingestion stream
    DOCUMENT_RECEIVED = "DOCUMENT_RECEIVED"
    DOCUMENT_DUPLICATE = "DOCUMENT_DUPLICATE"
    TEXT_EXTRACTED = "TEXT_EXTRACTED"
    TEXT_CLEANED = "TEXT_CLEANED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"


class TraceEvent(BaseModel):
    """Structured trace event emitted during pipeline execution."""

    seq: int
    type: EventType
    status: Literal["active", "ok", "warn", "error"]
    t_ms: int
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a pure JSON-serializable dictionary representation of event."""
        return {
            "seq": self.seq,
            "type": self.type.value if isinstance(self.type, Enum) else str(self.type),
            "status": self.status,
            "t_ms": self.t_ms,
            "label": self.label,
            "detail": self.detail,
        }


class EventEmitter:
    """
    Per-run event emitter maintaining sequence numbers and start time.
    MUST be instantiated locally inside pipeline.run() for run isolation.
    """

    def __init__(self) -> None:
        self._seq = 1
        self._t0 = time.perf_counter()

    @property
    def t_ms(self) -> int:
        """Measured elapsed milliseconds from run start (perf_counter)."""
        return int((time.perf_counter() - self._t0) * 1000)

    def emit(
        self,
        event_type: EventType,
        status: Literal["active", "ok", "warn", "error"],
        label: str,
        detail: dict[str, Any],
    ) -> TraceEvent:
        """Construct and return TraceEvent without yielding or performing I/O."""
        current_seq = self._seq
        self._seq += 1

        return TraceEvent(
            seq=current_seq,
            type=event_type,
            status=status,
            t_ms=self.t_ms,
            label=label,
            detail=detail,
        )
