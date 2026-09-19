from backend.app.models import Chunk, RetrievedChunk
from backend.generation.context_builder import build_generation_context


def _make_retrieved_chunk(
    chunk_id: str, doc: str = "doc1.pdf", page: int = 1, text: str = "Passage text"
) -> RetrievedChunk:
    c = Chunk(
        chunk_id=chunk_id,
        doc_id="sha123",
        document=doc,
        page=page,
        chunk_index=1,
        text=text,
        char_count=len(text),
        token_estimate=len(text) // 4 + 1,
    )
    return RetrievedChunk(chunk=c, score=0.88, rank=1, above_threshold=True)


def test_build_generation_context_preserves_metadata():
    chunks = [
        _make_retrieved_chunk(
            "docA__p001__c0001", "docA.pdf", 1, "First paragraph text."
        ),
        _make_retrieved_chunk(
            "docB__p003__c0005", "docB.pdf", 3, "Second paragraph text."
        ),
    ]
    question = "What are the requirements?"

    ctx = build_generation_context(question, chunks)

    assert ctx.question == question
    assert len(ctx.selected_chunks) == 2
    assert ctx.passages_count == 2
    assert ctx.selected_chunks[0].chunk.chunk_id == "docA__p001__c0001"
    assert ctx.selected_chunks[0].chunk.document == "docA.pdf"
    assert ctx.selected_chunks[0].chunk.page == 1
    assert ctx.selected_chunks[0].chunk.text == "First paragraph text."

    assert ctx.selected_chunks[1].chunk.chunk_id == "docB__p003__c0005"
    assert ctx.selected_chunks[1].chunk.document == "docB.pdf"
    assert ctx.selected_chunks[1].chunk.page == 3
    assert ctx.selected_chunks[1].chunk.text == "Second paragraph text."

    assert ctx.context_chars == len("First paragraph text.") + len(
        "Second paragraph text."
    )
    assert ctx.token_estimate > 0
    assert "docA__p001__c0001" in ctx.formatted_prompt
    assert "docB__p003__c0005" in ctx.formatted_prompt
