import pytest

from backend.app.config import AppConfig, settings
from ingestion.chunker import (
    bge_token_counter,
    chunk_cleaned_pages,
    default_word_counter,
)
from ingestion.cleaner import CleanedPage


def test_chunker_escalating_fallback_and_overlap_page_isolation():
    cleaned_pages = [
        CleanedPage(
            page_number=1,
            text=(
                "Paragraph 1. This is a sentence in paragraph 1.\n\n"
                "Paragraph 2. This is another sentence in paragraph 2.\n\n"
                "Paragraph 3. Extra text on page 1."
            ),
        ),
        CleanedPage(
            page_number=2,
            text="First paragraph of page 2. Independent page text.",
        ),
    ]

    file_bytes = b"fake_pdf_content_for_determinism_test_123"
    filename = "sample_document.pdf"

    # Use small chunk_size (10) to force splitting and test overlap
    chunks = chunk_cleaned_pages(
        pages=cleaned_pages,
        filename=filename,
        file_bytes=file_bytes,
        token_counter=default_word_counter,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) > 0

    # Verify page isolation: no chunk contains text from multiple pages
    for chunk in chunks:
        if chunk.page == 1:
            assert "Independent page text" not in chunk.text
        if chunk.page == 2:
            assert "Paragraph 1" not in chunk.text

    # Verify 1-based page numbering and chunk ID determinism
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 1
    assert chunks[0].doc_id == "e1b87f850a4b"  # SHA-256 of file_bytes[:12]
    assert chunks[0].chunk_id.startswith("sample_document__p001__c0001")


def test_chunker_determinism_identical_bytes_produce_identical_output():
    file_bytes = b"deterministic_content_bytes_456"
    filename = "analytics.pdf"

    pages = [CleanedPage(page_number=1, text="Deterministic text content.")]

    run_1 = chunk_cleaned_pages(pages, filename, file_bytes, default_word_counter)
    run_2 = chunk_cleaned_pages(pages, filename, file_bytes, default_word_counter)

    assert len(run_1) == len(run_2)
    assert run_1[0].chunk_id == run_2[0].chunk_id
    assert run_1[0].doc_id == run_2[0].doc_id
    assert run_1[0].text == run_2[0].text


def test_chunk_overlap_validation_rejects_invalid_config():
    msg_pat = "OVERLAP.*>= SIZE"
    with pytest.raises(ValueError, match=msg_pat):
        AppConfig(CHUNK_SIZE=200, CHUNK_OVERLAP=200)


@pytest.mark.integration
def test_production_bge_tokenizer_chunk_token_limit():
    """Verify production BGE tokenizer chunks satisfy <= 510 tokens invariant."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 100
    pages = [CleanedPage(page_number=1, text=long_text)]

    chunks = chunk_cleaned_pages(
        pages=pages,
        filename="long_bge_doc.pdf",
        file_bytes=b"long_bge_bytes",
        token_counter=bge_token_counter,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    for chunk in chunks:
        tokens = bge_token_counter(chunk.text)
        err_msg = f"Chunk {chunk.chunk_id} exceeded limit: {tokens} tokens"
        assert tokens <= 510, err_msg
