from backend.app.models import Chunk, RetrievedChunk
from backend.generation.prompt import GROUNDED_SYSTEM_PROMPT, format_grounded_prompt


def _make_retrieved_chunk(
    chunk_id: str, doc: str = "doc1.pdf", page: int = 1
) -> RetrievedChunk:
    c = Chunk(
        chunk_id=chunk_id,
        doc_id="sha123",
        document=doc,
        page=page,
        chunk_index=1,
        text="Sample text content for testing prompt.",
        char_count=39,
        token_estimate=8,
    )
    return RetrievedChunk(chunk=c, score=0.85, rank=1, above_threshold=True)


def test_grounded_prompt_instructions():
    assert "ONLY the provided knowledge-base context passages" in GROUNDED_SYSTEM_PROMPT
    assert "Do NOT use outside knowledge" in GROUNDED_SYSTEM_PROMPT
    assert (
        "The knowledge base does not contain enough information"
        in GROUNDED_SYSTEM_PROMPT
    )
    assert "[chunk_id]" in GROUNDED_SYSTEM_PROMPT


def test_format_grounded_prompt_includes_question_and_chunks():
    chunks = [
        _make_retrieved_chunk(
            "edge_computing_iot__p001__c0001", "edge_computing_iot.pdf", 1
        ),
        _make_retrieved_chunk(
            "edge_computing_iot__p002__c0003", "edge_computing_iot.pdf", 2
        ),
    ]

    prompt = format_grounded_prompt("Why does latency matter?", chunks)

    assert "Why does latency matter?" in prompt
    assert "edge_computing_iot__p001__c0001" in prompt
    assert "edge_computing_iot__p002__c0003" in prompt
    assert "edge_computing_iot.pdf (Page 1)" in prompt
    assert "edge_computing_iot.pdf (Page 2)" in prompt
    assert "Sample text content for testing prompt." in prompt
