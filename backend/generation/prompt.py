from __future__ import annotations

from backend.app.models import RetrievedChunk

GROUNDED_SYSTEM_PROMPT = """You are a precise, evidence-grounded AI knowledge assistant.

STRICT INSTRUCTIONS:
1. Answer the user's question using ONLY the provided knowledge-base context passages
 below.
2. Do NOT use outside knowledge, external facts, or unmentioned assumptions.
3. If the provided context does NOT contain enough information to answer the
 question, state clearly: "The knowledge base does not contain enough information to
 answer this question."
4. Do NOT invent facts or hallucinate details not explicitly supported by the context.
5. You MUST cite your statements using the exact chunk IDs provided in brackets,
 e.g. [chunk_id].
6. Do NOT cite any chunk ID that is not listed in the supplied context.
7. Keep your answer concise, accurate, and directly focused on the user's question."""


def format_grounded_prompt(question: str, selected_chunks: list[RetrievedChunk]) -> str:
    """
    Format a grounded prompt containing system instructions, structured context
    passages with stable chunk IDs, and the user question.
    """
    passages_formatted = []
    for c in selected_chunks:
        chunk_obj = c.chunk
        passages_formatted.append(
            f"--- PASSAGE [{chunk_obj.chunk_id}] ---\n"
            f"Document: {chunk_obj.document} (Page {chunk_obj.page})\n"
            f"Content: {chunk_obj.text}\n"
        )

    context_str = "\n".join(passages_formatted)

    prompt = (
        f"{GROUNDED_SYSTEM_PROMPT}\n\n"
        f"=== SUPPLIED KNOWLEDGE-BASE CONTEXT ===\n"
        f"{context_str}\n"
        f"=== USER QUESTION ===\n"
        f"{question}\n\n"
        f"=== ANSWER (with [chunk_id] citations) ==="
    )
    return prompt
