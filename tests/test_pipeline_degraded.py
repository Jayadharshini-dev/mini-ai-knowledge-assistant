from __future__ import annotations

import pytest

from backend.app.errors import ErrorCode
from backend.generation.base import RawGenerationResult
from backend.generation.fake_provider import FakeLLMProvider
from backend.rag.events import EventType
from backend.rag.outcome import collect_run
from backend.rag.pipeline import RagPipeline
from tests.conftest import assert_valid_event_stream


@pytest.mark.asyncio
async def test_pipeline_degraded_no_provider_configured(mock_pipeline_components):
    # Use FakeLLMProvider with is_available_override=False
    mock_pipeline_components["provider"] = FakeLLMProvider(is_available_override=False)

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
        EventType.GENERATION_SKIPPED,
        EventType.COMPLETE,
    ]
    assert event_types == expected_types

    assert outcome.degraded == "no_provider"
    assert outcome.answer is None
    assert outcome.citations == []
    assert len(outcome.selected_chunks) > 0


@pytest.mark.asyncio
async def test_pipeline_degraded_transient_rate_limit(mock_pipeline_components):
    class RateLimitedProvider:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        def is_available(self) -> bool:
            return True

        def generate(self, context):
            return RawGenerationResult(
                text=None,
                provider=self.provider_name,
                model=self.model_name,
                status="failed",
                error_code=ErrorCode.PROVIDER_RATE_LIMITED,
                error_message="429 Rate limit exceeded",
                elapsed_ms=50,
            )

    mock_pipeline_components["provider"] = RateLimitedProvider()

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Rate limited query"))
    events, outcome = collected.events, collected.outcome

    assert_valid_event_stream(events)

    assert outcome.degraded == "PROVIDER_RATE_LIMITED"
    assert outcome.answer is None
    assert outcome.citations == []


@pytest.mark.asyncio
async def test_pipeline_provider_auth_failure_emits_error(mock_pipeline_components):
    class AuthFailedProvider:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        def is_available(self) -> bool:
            return True

        def generate(self, context):
            return RawGenerationResult(
                text=None,
                provider=self.provider_name,
                model=self.model_name,
                status="unavailable",
                error_code=ErrorCode.PROVIDER_AUTH,
                error_message="Invalid GEMINI_API_KEY",
                elapsed_ms=10,
            )

    mock_pipeline_components["provider"] = AuthFailedProvider()

    pipeline = RagPipeline(**mock_pipeline_components)
    collected = await collect_run(pipeline.run("Auth failed query"))
    events, outcome = collected.events, collected.outcome

    assert_valid_event_stream(events)
    assert events[-1].type == EventType.ERROR
    assert events[-1].detail["code"] == "PROVIDER_AUTH"
    assert outcome.degraded == "error"
