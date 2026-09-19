from __future__ import annotations

import json

import numpy as np
import pytest

from backend.rag.events import EventEmitter, EventType, TraceEvent
from backend.rag.pipeline import _to_json_safe


def test_event_type_enum_values():
    """Verify EventType enum contains all required contract query stream event types."""
    expected = {
        "QUERY_RECEIVED",
        "RETRIEVAL_STARTED",
        "RETRIEVAL_COMPLETED",
        "EVIDENCE_SELECTED",
        "ABSTAINED",
        "CONTEXT_BUILT",
        "GENERATION_STARTED",
        "GENERATION_COMPLETED",
        "GENERATION_SKIPPED",
        "COMPLETE",
        "ERROR",
    }
    actual = {e.value for e in EventType if e.value in expected}
    assert actual == expected


def test_trace_event_to_dict():
    """Verify TraceEvent converts cleanly to dict and serializes to JSON."""
    event = TraceEvent(
        seq=1,
        status="ok",
        t_ms=12,
        label="Query received",
        type=EventType.QUERY_RECEIVED,
        detail={"question": "What is IoT?"},
    )
    d = event.to_dict()
    assert d["seq"] == 1
    assert d["t_ms"] == 12
    assert d["type"] == "QUERY_RECEIVED"
    assert d["status"] == "ok"
    assert d["label"] == "Query received"
    assert d["detail"] == {"question": "What is IoT?"}

    json_str = json.dumps(d)
    assert "QUERY_RECEIVED" in json_str


def test_event_emitter_scoping_and_sequence():
    """Verify EventEmitter starts sequence at 1 and emits monotonic t_ms timestamps."""
    emitter = EventEmitter()
    assert emitter._seq == 1

    e1 = emitter.emit(EventType.QUERY_RECEIVED, "ok", "Label 1", {"q": "test"})
    assert e1.seq == 1
    assert e1.t_ms >= 0

    e2 = emitter.emit(EventType.RETRIEVAL_STARTED, "active", "Label 2", {})
    assert e2.seq == 2
    assert e2.t_ms >= e1.t_ms

    e3 = emitter.emit(EventType.COMPLETE, "ok", "Label 3", {"status": "ok"})
    assert e3.seq == 3
    assert e3.t_ms >= e2.t_ms


def test_to_json_safe_strict_conversions():
    """Verify _to_json_safe handles primitives, tuples, and raises TypeError."""
    # Primitives
    assert _to_json_safe(42) == 42
    assert _to_json_safe("text") == "text"
    assert _to_json_safe(True) is True
    assert _to_json_safe(None) is None

    # Tuple converted to list (JSON array)
    assert _to_json_safe((1, "two", 3.0)) == [1, "two", 3.0]

    # Numpy scalar conversion
    np_val = np.float32(0.85)
    converted_np = _to_json_safe(np_val)
    assert isinstance(converted_np, float)
    assert converted_np == float(np_val)

    # Unsupported type MUST raise TypeError
    class DummyClass:
        pass

    with pytest.raises(TypeError, match="is not JSON serializable"):
        _to_json_safe(DummyClass())
