from backend.app.models import Chunk, RetrievedChunk
from evaluation.metrics import (
    hit_at_k,
    is_chunk_relevant,
    matches_evidence,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def _make_retrieved_chunk(
    doc: str, page: int, rank: int = 1, score: float = 0.8
) -> RetrievedChunk:
    chunk_obj = Chunk(
        chunk_id=f"doc__p{page:03d}__c{rank:04d}",
        doc_id="sha123",
        document=doc,
        page=page,
        chunk_index=rank,
        text="Sample passage text",
        char_count=19,
        token_estimate=4,
    )
    return RetrievedChunk(
        chunk=chunk_obj,
        score=score,
        rank=rank,
        above_threshold=score >= 0.32,
    )


def test_matches_evidence():
    chunk = _make_retrieved_chunk(doc="Edge_Computing_IoT.pdf", page=2)

    assert (
        matches_evidence(chunk, {"document": "edge_computing_iot.pdf", "page": 2})
        is True
    )
    assert (
        matches_evidence(chunk, {"document": "edge_computing_iot.pdf", "page": 3})
        is False
    )
    assert matches_evidence(chunk, {"document": "other_doc.pdf", "page": 2}) is False


def test_is_chunk_relevant():
    chunk = _make_retrieved_chunk(doc="campus_resource_management.pdf", page=5)
    evidence = [
        {"document": "edge_computing_iot.pdf", "page": 1},
        {"document": "campus_resource_management.pdf", "page": 5},
    ]

    assert is_chunk_relevant(chunk, evidence) is True
    assert (
        is_chunk_relevant(
            chunk, [{"document": "campus_resource_management.pdf", "page": 4}]
        )
        is False
    )


def test_hit_at_k():
    retrieved = [
        _make_retrieved_chunk(doc="docA.pdf", page=1, rank=1),
        _make_retrieved_chunk(doc="docB.pdf", page=2, rank=2),
        _make_retrieved_chunk(doc="docC.pdf", page=3, rank=3),
    ]
    evidence = [{"document": "docB.pdf", "page": 2}]

    assert hit_at_k(retrieved, evidence, k=1) == 0.0
    assert hit_at_k(retrieved, evidence, k=2) == 1.0
    assert hit_at_k(retrieved, [], k=2) == 0.0


def test_recall_at_k():
    retrieved = [
        _make_retrieved_chunk(doc="docA.pdf", page=1, rank=1),
        _make_retrieved_chunk(doc="docB.pdf", page=2, rank=2),
    ]
    evidence = [
        {"document": "docA.pdf", "page": 1},
        {"document": "docB.pdf", "page": 2},
        {"document": "docC.pdf", "page": 3},
    ]

    assert recall_at_k(retrieved, evidence, k=1) == 1 / 3
    assert recall_at_k(retrieved, evidence, k=2) == 2 / 3
    assert recall_at_k(retrieved, [], k=2) == 0.0


def test_recall_at_5_evaluates_5_candidates_independently():
    # Construct 5 retrieved candidates where match is at rank 5
    retrieved = [
        _make_retrieved_chunk(doc="docA.pdf", page=1, rank=1),
        _make_retrieved_chunk(doc="docB.pdf", page=2, rank=2),
        _make_retrieved_chunk(doc="docC.pdf", page=3, rank=3),
        _make_retrieved_chunk(doc="docD.pdf", page=4, rank=4),
        _make_retrieved_chunk(doc="target.pdf", page=10, rank=5),
    ]
    evidence = [{"document": "target.pdf", "page": 10}]

    # Production TOP_K=4 would miss rank 5 (recall=0.0)
    assert recall_at_k(retrieved[:4], evidence, k=5) == 0.0
    # Evaluation with 5 retrieved candidates captures rank 5 (recall=1.0)
    assert recall_at_k(retrieved, evidence, k=5) == 1.0


def test_reciprocal_rank():
    retrieved = [
        _make_retrieved_chunk(doc="docA.pdf", page=1, rank=1),
        _make_retrieved_chunk(doc="docB.pdf", page=2, rank=2),
    ]
    evidence = [{"document": "docB.pdf", "page": 2}]

    assert reciprocal_rank(retrieved, evidence) == 0.5
    assert reciprocal_rank(retrieved, [{"document": "docC.pdf", "page": 1}]) == 0.0
    assert reciprocal_rank(retrieved, []) == 0.0


def test_mean_reciprocal_rank():
    rr_scores = [1.0, 0.5, 0.0, 0.3333333333333333]
    assert mean_reciprocal_rank(rr_scores) == (1.0 + 0.5 + 0.0 + 1 / 3) / 4
    assert mean_reciprocal_rank([]) == 0.0
