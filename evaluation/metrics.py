from __future__ import annotations

from typing import Any

from backend.app.models import RetrievedChunk


def matches_evidence(chunk: RetrievedChunk, expected_pair: dict[str, Any]) -> bool:
    """Return True if chunk document and page match expected evidence pair."""
    exp_doc = str(expected_pair.get("document", "")).strip().lower()
    exp_page = int(expected_pair.get("page", 0))

    chunk_doc = chunk.chunk.document.strip().lower()
    chunk_page = chunk.chunk.page

    return chunk_doc == exp_doc and chunk_page == exp_page


def is_chunk_relevant(
    chunk: RetrievedChunk, expected_evidence: list[dict[str, Any]]
) -> bool:
    """Return True if chunk matches any expected evidence pair in the list."""
    return any(matches_evidence(chunk, exp) for exp in expected_evidence)


def hit_at_k(
    retrieved: list[RetrievedChunk],
    expected_evidence: list[dict[str, Any]],
    k: int,
) -> float:
    """
    Hit@K metric: returns 1.0 if at least one chunk in retrieved[:k] matches ground
    truth, else 0.0.
    """
    if not expected_evidence:
        return 0.0

    top_k_chunks = retrieved[:k]
    for chunk in top_k_chunks:
        if is_chunk_relevant(chunk, expected_evidence):
            return 1.0
    return 0.0


def recall_at_k(
    retrieved: list[RetrievedChunk],
    expected_evidence: list[dict[str, Any]],
    k: int,
) -> float:
    """
    Recall@K metric: fraction of expected ground-truth evidence pairs present in
    retrieved[:k].
    """
    if not expected_evidence:
        return 0.0

    top_k_chunks = retrieved[:k]
    matched_count = 0

    for exp in expected_evidence:
        if any(matches_evidence(c, exp) for c in top_k_chunks):
            matched_count += 1

    return matched_count / len(expected_evidence)


def reciprocal_rank(
    retrieved: list[RetrievedChunk],
    expected_evidence: list[dict[str, Any]],
) -> float:
    """
    Reciprocal Rank (RR): 1 / rank of the first relevant chunk retrieved, or 0.0 if
    no match.
    """
    if not expected_evidence:
        return 0.0

    for idx, chunk in enumerate(retrieved):
        if is_chunk_relevant(chunk, expected_evidence):
            rank = idx + 1
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """
    Mean Reciprocal Rank (MRR): average of reciprocal ranks across a set of queries.
    """
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)
