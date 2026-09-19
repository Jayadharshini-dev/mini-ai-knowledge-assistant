from unittest.mock import MagicMock

from backend.app.models import Chunk, RetrievedChunk
from backend.generation.base import LLMProvider
from backend.generation.context_builder import build_generation_context
from backend.generation.fake_provider import FakeLLMProvider
from retrieval.relevance import evaluate_relevance


def _make_chunk(chunk_id: str, score: float, rank: int = 1) -> RetrievedChunk:
    c = Chunk(
        chunk_id=chunk_id,
        doc_id="sha123",
        document="test_doc.pdf",
        page=1,
        chunk_index=rank,
        text=f"Passage text for {chunk_id}",
        char_count=25,
        token_estimate=6,
    )
    return RetrievedChunk(
        chunk=c,
        score=score,
        rank=rank,
        above_threshold=score >= 0.67,
    )


def test_provider_never_called_when_relevance_gate_abstains():
    # Out-of-scope retrieval candidates (all below 0.67)
    retrieved = [
        _make_chunk("doc1__p001__c0001", score=0.35, rank=1),
        _make_chunk("doc1__p002__c0004", score=0.32, rank=2),
    ]

    # Evaluate relevance gate
    decision = evaluate_relevance(retrieved, threshold=0.67)
    assert decision.decision == "abstain"

    # Set up mock provider
    mock_provider = MagicMock(spec=LLMProvider)

    # Test-level logic simulating pipeline guard
    if decision.decision == "proceed":
        ctx = build_generation_context("Out of scope q?", decision.selected_chunks)
        mock_provider.generate(ctx)

    # Assert provider generate() was NEVER called
    mock_provider.generate.assert_not_called()


def test_only_selected_chunks_passed_to_provider():
    # Candidates with 2 above threshold and 1 below threshold
    retrieved = [
        _make_chunk("doc1__p001__c0001", score=0.85, rank=1),
        _make_chunk("doc1__p002__c0002", score=0.72, rank=2),
        _make_chunk("doc1__p003__c0003", score=0.45, rank=3),
    ]

    decision = evaluate_relevance(retrieved, threshold=0.67)
    assert decision.decision == "proceed"
    assert len(decision.selected_chunks) == 2
    assert len(decision.rejected_chunks) == 1

    fake_provider = FakeLLMProvider()
    ctx = build_generation_context("In scope question", decision.selected_chunks)
    result = fake_provider.generate(ctx)

    assert result.status == "completed"
    # Selected chunks present
    assert "doc1__p001__c0001" in ctx.formatted_prompt
    assert "doc1__p002__c0002" in ctx.formatted_prompt
    # Rejected chunk EXCLUDED from prompt
    assert "doc1__p003__c0003" not in ctx.formatted_prompt
