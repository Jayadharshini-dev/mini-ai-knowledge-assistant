from __future__ import annotations

from backend.app.models import RetrievedChunk
from backend.generation.base import GenerationContext
from backend.generation.prompt import format_grounded_prompt


def build_generation_context(
    question: str, selected_chunks: list[RetrievedChunk]
) -> GenerationContext:
    """
    Build a structured GenerationContext from a user question and selected evidence
    chunks.
    Preserves all metadata required for citation validation (chunk_id, document,
    page, text).
    """
    formatted_prompt = format_grounded_prompt(question, selected_chunks)

    context_chars = sum(len(c.chunk.text) for c in selected_chunks)
    token_estimate = sum(c.chunk.token_estimate for c in selected_chunks)

    return GenerationContext(
        question=question,
        selected_chunks=selected_chunks,
        formatted_prompt=formatted_prompt,
        passages_count=len(selected_chunks),
        context_chars=context_chars,
        token_estimate=token_estimate,
        truncated=False,
    )
