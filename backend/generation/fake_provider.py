from __future__ import annotations

import time
from typing import Optional

from backend.generation.base import GenerationContext, LLMProvider, RawGenerationResult


class FakeLLMProvider(LLMProvider):
    """
    Deterministic offline fake LLM provider for unit tests and offline execution.
    Does not make real network calls.
    """

    def __init__(
        self,
        custom_response: Optional[str] = None,
        is_available_override: bool = True,
        inject_invalid_citation: bool = False,
    ):
        self._custom_response = custom_response
        self._available = is_available_override
        self._inject_invalid_citation = inject_invalid_citation

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-gemini-2.5-flash"

    def is_available(self) -> bool:
        return self._available

    def generate(self, context: GenerationContext) -> RawGenerationResult:
        """Return deterministic generated text based on supplied context."""
        start_time = time.perf_counter()

        if not self.is_available():
            return RawGenerationResult(
                text=None,
                provider=self.provider_name,
                model=self.model_name,
                status="unavailable",
                error_message="Fake LLM Provider set to unavailable.",
                elapsed_ms=0,
            )

        if self._custom_response is not None:
            text = self._custom_response
        else:
            # Deterministic response citing selected chunks
            citations_text = []
            for c in context.selected_chunks:
                citations_text.append(f"[{c.chunk.chunk_id}]")

            if self._inject_invalid_citation:
                citations_text.append("[nonexistent_chunk_id_999]")

            cites_str = " ".join(citations_text)
            text = (
                f"Based on the provided passages, here is the answer to"
                f" '{context.question}'. Evidence: {cites_str}"
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return RawGenerationResult(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            status="completed",
            elapsed_ms=elapsed_ms,
        )
