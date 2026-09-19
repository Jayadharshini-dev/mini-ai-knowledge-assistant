import pytest

from backend.app.errors import AppException, ErrorCode
from ingestion.loader import load_pdf_pages
from tests.fixtures.make_fixture_pdf import (
    create_scanned_test_pdf_bytes,
    create_test_pdf_bytes,
)


def test_loader_extracts_known_text_with_1_based_pages():
    pages_text = [
        "First page content introduction.",
        "Second page content methodology.",
        "Third page content results.",
    ]
    pdf_bytes = create_test_pdf_bytes(pages_text)

    pages = load_pdf_pages(pdf_bytes, "test_doc.pdf")

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[2].page_number == 3

    assert "First page content" in pages[0].full_text
    assert "Second page content" in pages[1].full_text
    assert "Third page content" in pages[2].full_text


def test_loader_scanned_pdf_raises_no_extractable_text():
    pdf_bytes = create_scanned_test_pdf_bytes()

    with pytest.raises(AppException) as exc_info:
        load_pdf_pages(pdf_bytes, "scanned.pdf")

    assert exc_info.value.code == ErrorCode.NO_EXTRACTABLE_TEXT
    assert exc_info.value.status_code == 422


def test_loader_encrypted_pdf_raises_encrypted_error():
    pdf_bytes = create_test_pdf_bytes(["Protected content"], encrypted=True)

    with pytest.raises(AppException) as exc_info:
        load_pdf_pages(pdf_bytes, "encrypted.pdf")

    assert exc_info.value.code == ErrorCode.ENCRYPTED_PDF
    assert exc_info.value.status_code == 400


def test_loader_invalid_pdf_bytes_raises_invalid_pdf():
    with pytest.raises(AppException) as exc_info:
        load_pdf_pages(b"NOT_A_PDF_HEADER", "corrupt.pdf")

    assert exc_info.value.code == ErrorCode.INVALID_PDF
    assert exc_info.value.status_code == 400


def test_loader_page_count_matches_pymupdf():
    import fitz

    pages_text = ["Page 1 text", "Page 2 text", "Page 3 text", "Page 4 text"]
    pdf_bytes = create_test_pdf_bytes(pages_text)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    expected_page_count = len(doc)

    pages = load_pdf_pages(pdf_bytes, "multi_page.pdf")

    assert len(pages) == expected_page_count == 4
    assert max(p.page_number for p in pages) == expected_page_count
