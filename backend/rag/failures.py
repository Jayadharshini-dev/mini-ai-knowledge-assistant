from __future__ import annotations

from typing import Any, Optional

from backend.app.errors import AppException, ErrorCode


def map_exception_to_error_detail(
    exc: Exception, stage: Optional[str] = None
) -> dict[str, Any]:
    """
    Map pipeline exceptions to structured JSON-serializable ERROR event details.
    Uses typed AppException when available, or pipeline stage context.
    Never exposes raw internal exception details or tracebacks to the client envelope.
    """
    if isinstance(exc, AppException):
        return {
            "code": exc.code.value,
            "message": exc.message,
            "retryable": exc.retryable,
        }

    # Map based on stage context when available
    if stage == "retrieve":
        code = ErrorCode.RETRIEVAL_FAILED.value
        user_message = "An error occurred during vector retrieval."
    elif stage == "generate":
        code = ErrorCode.PROVIDER_UNAVAILABLE.value
        user_message = "An error occurred during LLM generation."
    else:
        code = ErrorCode.INTERNAL_ERROR.value
        user_message = "An internal error occurred during pipeline execution."

    return {
        "code": code,
        "message": user_message,
        "retryable": False,
    }
