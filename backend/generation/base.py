from __future__ import annotations

from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

from backend.app.errors import ErrorCode
from backend.app.models import RetrievedChunk


class GenerationContext(BaseModel):
    """Structured context passed to an LLM provider for grounded generation."""

    question: str
    selected_chunks: list[RetrievedChunk] = Field(default_factory=list)
    formatted_prompt: str
    passages_count: int
    context_chars: int
    token_estimate: int


class RawGenerationResult(BaseModel):
    """Raw, un-validated output returned directly from an LLM provider."""

    text: Optional[str] = None
    provider: str
    model: str
    status: Literal["completed", "failed", "unavailable"]
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
    elapsed_ms: int = 0


class LLMProvider(Protocol):
    """Protocol interface defining the minimum generation operation."""

    @property
    def provider_name(self) -> str:
        """Provider identifier string (e.g. 'gemini', 'fake')."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier string (e.g. 'gemini-2.5-flash')."""
        ...

    def is_available(self) -> bool:
        """Return True if provider is configured and available for generation calls."""
        ...

    def generate(self, context: GenerationContext) -> RawGenerationResult:
        """Generate response text from the supplied grounded context."""
        ...
