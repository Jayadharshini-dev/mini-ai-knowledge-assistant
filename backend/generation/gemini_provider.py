from __future__ import annotations

import time
from typing import Optional

from backend.app.config import settings
from backend.app.errors import ErrorCode
from backend.generation.base import GenerationContext, LLMProvider, RawGenerationResult


class GeminiLLMProvider(LLMProvider):
    """
    Real Gemini LLM Provider implementation using the official google-genai SDK.
    Performs raw model generation only; does not contain application-level citation
    validation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self._api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self._model_name = model_name or settings.LLM_MODEL

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Return True if GEMINI_API_KEY is non-empty."""
        return bool(self._api_key and self._api_key.strip())

    def generate(self, context: GenerationContext) -> RawGenerationResult:
        """
        Generate raw text from Gemini given the supplied grounded context.
        Returns a structured RawGenerationResult.
        """
        if not self.is_available():
            return RawGenerationResult(
                text=None,
                provider=self.provider_name,
                model=self.model_name,
                status="unavailable",
                error_code=ErrorCode.PROVIDER_UNAVAILABLE,
                error_message="Gemini API key is missing or not configured.",
                elapsed_ms=0,
            )

        start_time = time.perf_counter()
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(
                    timeout=settings.LLM_REQUEST_TIMEOUT * 1000
                ),
            )
            config = types.GenerateContentConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=context.formatted_prompt,
                config=config,
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Safely inspect response candidates and finish reasons
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return RawGenerationResult(
                    text=None,
                    provider=self.provider_name,
                    model=self.model_name,
                    status="failed",
                    error_code=ErrorCode.PROVIDER_UNAVAILABLE,
                    error_message="Gemini response contained no output candidates.",
                    elapsed_ms=elapsed_ms,
                )

            candidate = candidates[0]
            finish_reason = str(getattr(candidate, "finish_reason", "")).upper()
            if any(
                block in finish_reason for block in ["SAFETY", "RECITATION", "BLOCK"]
            ):
                return RawGenerationResult(
                    text=None,
                    provider=self.provider_name,
                    model=self.model_name,
                    status="failed",
                    error_code=ErrorCode.PROVIDER_UNAVAILABLE,
                    error_message=f"Gemini generation blocked: {finish_reason}",
                    elapsed_ms=elapsed_ms,
                )

            text_out = getattr(response, "text", "") or ""

            return RawGenerationResult(
                text=text_out,
                provider=self.provider_name,
                model=self.model_name,
                status="completed",
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            exc_str = str(exc).lower()

            if "401" in exc_str or "auth" in exc_str or "key" in exc_str:
                err_code = ErrorCode.PROVIDER_AUTH
            elif "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
                err_code = ErrorCode.PROVIDER_RATE_LIMITED
            else:
                err_code = ErrorCode.PROVIDER_UNAVAILABLE

            return RawGenerationResult(
                text=None,
                provider=self.provider_name,
                model=self.model_name,
                status="failed",
                error_code=err_code,
                error_message=f"Gemini API request failed: {exc}",
                elapsed_ms=elapsed_ms,
            )
