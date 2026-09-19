from backend.app.models import Chunk, RetrievedChunk
from backend.generation.citation_validator import validate_citations


def _make_retrieved_chunk(chunk_id: str, doc: str, page: int) -> RetrievedChunk:
    c = Chunk(
        chunk_id=chunk_id,
        doc_id="sha123",
        document=doc,
        page=page,
        chunk_index=1,
        text="Sample passage text.",
        char_count=20,
        token_estimate=5,
    )
    return RetrievedChunk(chunk=c, score=0.85, rank=1, above_threshold=True)


def test_validate_citations_retains_valid_chunk_ids():
    chunks = [
        _make_retrieved_chunk(
            "edge_computing_iot__p001__c0001", "edge_computing_iot.pdf", 1
        ),
        _make_retrieved_chunk(
            "edge_computing_iot__p002__c0003", "edge_computing_iot.pdf", 2
        ),
    ]

    raw_text = (
        "Latency is crucial [edge_computing_iot__p001__c0001]. "
        "Schema is enforced at gateway [edge_computing_iot__p002__c0003]."
    )

    result = validate_citations(raw_text, chunks)

    assert result.citations_emitted == 2
    assert result.citations_dropped == 0
    assert len(result.valid_citations) == 2
    assert result.valid_citations[0].n == 1
    assert result.valid_citations[0].chunk_id == "edge_computing_iot__p001__c0001"
    assert result.valid_citations[0].document == "edge_computing_iot.pdf"
    assert result.valid_citations[0].page == 1

    assert result.valid_citations[1].n == 2
    assert result.valid_citations[1].chunk_id == "edge_computing_iot__p002__c0003"
    assert result.valid_citations[1].page == 2


def test_validate_citations_drops_nonexistent_ids():
    chunks = [
        _make_retrieved_chunk(
            "edge_computing_iot__p001__c0001", "edge_computing_iot.pdf", 1
        ),
    ]

    raw_text = (
        "Latency is crucial [edge_computing_iot__p001__c0001]. "
        "Unrelated fact [fake_doc__p999__c9999]."
    )

    result = validate_citations(raw_text, chunks)

    assert result.citations_emitted == 2
    assert result.citations_dropped == 1
    assert len(result.valid_citations) == 1
    assert result.valid_citations[0].chunk_id == "edge_computing_iot__p001__c0001"
    assert result.invalid_citation_ids == ["fake_doc__p999__c9999"]
    assert "fake_doc__p999__c9999" not in result.clean_text
    assert (
        result.clean_text
        == "Latency is crucial [edge_computing_iot__p001__c0001]. Unrelated fact."
    )


def test_validate_citations_strips_invalid_markers_from_clean_text():
    chunks = [
        _make_retrieved_chunk("valid_doc__p001__c0001", "valid_doc.pdf", 1),
    ]

    raw_text = (
        "Valid point [valid_doc__p001__c0001]. Invalid point [nonexistent_chunk_999]."
    )

    result = validate_citations(raw_text, chunks)

    assert result.citations_emitted == 2
    assert result.citations_dropped == 1
    assert "nonexistent_chunk_999" not in result.clean_text
    assert result.clean_text == "Valid point [valid_doc__p001__c0001]. Invalid point."


def test_validate_citations_empty_text():
    chunks = [_make_retrieved_chunk("doc1__p001__c0001", "doc1.pdf", 1)]
    result = validate_citations("", chunks)

    assert result.clean_text == ""
    assert result.valid_citations == []
    assert result.citations_emitted == 0
    assert result.citations_dropped == 0
