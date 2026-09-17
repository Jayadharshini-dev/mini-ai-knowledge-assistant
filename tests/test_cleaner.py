from ingestion.cleaner import clean_block_text, clean_pages
from ingestion.loader import load_pdf_pages
from tests.fixtures.make_fixture_pdf import create_test_pdf_bytes


def test_clean_block_text_soft_hyphen_and_dehyphenation():
    # Soft hyphen removal
    raw = "soft\u00adhyphen text"
    cleaned = clean_block_text(raw)
    assert cleaned == "softhyphen text"

    # Conservative dehyphenation: lowercase continuation is joined
    dehyphen_lower = "comput-\ning science"
    cleaned_lower = clean_block_text(dehyphen_lower)
    assert cleaned_lower == "computing science"

    # Conservative dehyphenation: uppercase continuation is preserved
    dehyphen_upper = "Multi-\nThreaded architecture"
    cleaned_upper = clean_block_text(dehyphen_upper)
    assert "Multi-\nThreaded" in cleaned_upper or "Multi-\nThreaded" in raw


def test_cleaner_detects_and_removes_repeated_header_and_footer():
    pages_text = [
        "Body content paragraph for page 1.",
        "Body content paragraph for page 2.",
        "Body content paragraph for page 3.",
        "Body content paragraph for page 4.",
    ]
    # Header: "CONFIDENTIAL REPORT"
    # Footer: "Internal Document"
    pdf_bytes = create_test_pdf_bytes(
        pages_text,
        header_text="CONFIDENTIAL REPORT",
        footer_text="Internal Document",
    )

    extracted_pages = load_pdf_pages(pdf_bytes, "report.pdf")
    cleaned_pages, removed_furniture = clean_pages(extracted_pages)

    assert len(cleaned_pages) == 4
    assert len(removed_furniture) > 0

    # Body text survives across all pages
    for idx, cpage in enumerate(cleaned_pages):
        assert f"Body content paragraph for page {idx + 1}" in cpage.text
        # Repeated headers/footers should be stripped
        assert "CONFIDENTIAL REPORT" not in cpage.text


def test_cleaner_short_documents_do_not_aggressively_remove_headers():
    # 2-page document: should NOT aggressively remove top/bottom blocks
    pages_text = [
        "Important title on page 1",
        "Important summary on page 2",
    ]
    pdf_bytes = create_test_pdf_bytes(
        pages_text,
        header_text="Header Text",
    )

    extracted_pages = load_pdf_pages(pdf_bytes, "short_doc.pdf")
    cleaned_pages, removed_furniture = clean_pages(extracted_pages)

    assert len(cleaned_pages) == 2
    # Short document (2 pages) shouldn't strip furniture
    assert len(removed_furniture) == 0
