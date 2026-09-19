from backend.app.models import Chunk, RetrievedChunk
from retrieval.relevance import evaluate_relevance


def _make_retrieved_chunk(
    rank: int, score: float, doc: str = "doc.pdf", page: int = 1
) -> RetrievedChunk:
    chunk_obj = Chunk(
        chunk_id=f"doc__p{page:03d}__c{rank:04d}",
        doc_id="sha123",
        document=doc,
        page=page,
        chunk_index=rank,
        text="Sample passage",
        char_count=14,
        token_estimate=3,
    )
    return RetrievedChunk(
        chunk=chunk_obj,
        score=score,
        rank=rank,
        above_threshold=score >= 0.32,
    )


def test_evaluate_relevance_above_threshold_proceeds():
    chunks = [
        _make_retrieved_chunk(rank=1, score=0.85),
        _make_retrieved_chunk(rank=2, score=0.40),
        _make_retrieved_chunk(rank=3, score=0.20),
    ]

    decision = evaluate_relevance(chunks, threshold=0.32)

    assert decision.decision == "proceed"
    assert decision.top_score == 0.85
    assert len(decision.selected_chunks) == 2
    assert len(decision.rejected_chunks) == 1
    assert decision.reason is None


def test_evaluate_relevance_below_threshold_abstains():
    chunks = [
        _make_retrieved_chunk(rank=1, score=0.25),
        _make_retrieved_chunk(rank=2, score=0.15),
    ]

    decision = evaluate_relevance(chunks, threshold=0.32)

    assert decision.decision == "abstain"
    assert decision.top_score == 0.25
    assert len(decision.selected_chunks) == 0
    assert len(decision.rejected_chunks) == 2
    assert decision.reason == "no_passage_above_threshold"


def test_evaluate_relevance_exact_boundary():
    chunks = [_make_retrieved_chunk(rank=1, score=0.32)]

    decision = evaluate_relevance(chunks, threshold=0.32)

    assert decision.decision == "proceed"
    assert len(decision.selected_chunks) == 1
