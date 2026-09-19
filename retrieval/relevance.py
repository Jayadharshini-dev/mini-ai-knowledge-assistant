from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.app.models import RetrievedChunk


class RelevanceDecision(BaseModel):
    decision: Literal["proceed", "abstain"]
    threshold: float
    top_score: float
    selected_chunks: list[RetrievedChunk] = Field(default_factory=list)
    rejected_chunks: list[RetrievedChunk] = Field(default_factory=list)
    reason: Optional[str] = None
    message: str


def evaluate_relevance(
    chunks: list[RetrievedChunk],
    threshold: float,
) -> RelevanceDecision:
    """
    Pure, deterministic function to evaluate whether retrieved chunks meet
    the relevance threshold.
    Separates chunks into selected (score >= threshold) and rejected
    (score < threshold).
    Returns a RelevanceDecision indicating whether to proceed or abstain.
    """
    if not chunks:
        return RelevanceDecision(
            decision="abstain",
            threshold=threshold,
            top_score=0.0,
            selected_chunks=[],
            rejected_chunks=[],
            reason="no_passage_above_threshold",
            message=(
                "The knowledge base does not contain information relevant to this"
                " question."
            ),
        )

    top_score = float(chunks[0].score)
    selected = [c for c in chunks if c.score >= threshold]
    rejected = [c for c in chunks if c.score < threshold]

    if selected:
        return RelevanceDecision(
            decision="proceed",
            threshold=threshold,
            top_score=top_score,
            selected_chunks=selected,
            rejected_chunks=rejected,
            reason=None,
            message=(
                f"Found {len(selected)} chunk(s) clearing relevance threshold"
                f" {threshold:.2f}."
            ),
        )
    else:
        return RelevanceDecision(
            decision="abstain",
            threshold=threshold,
            top_score=top_score,
            selected_chunks=[],
            rejected_chunks=rejected,
            reason="no_passage_above_threshold",
            message=(
                "The knowledge base does not contain information relevant to this"
                " question."
            ),
        )
