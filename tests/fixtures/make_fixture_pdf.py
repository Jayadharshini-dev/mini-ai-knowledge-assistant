from __future__ import annotations

import fitz


def create_test_pdf_bytes(
    pages_text: list[str],
    header_text: str | None = None,
    footer_text: str | None = None,
    encrypted: bool = False,
    password: str = "secret",
) -> bytes:
    """
    Generate a deterministic PDF in memory using PyMuPDF (fitz) for unit tests.
    Supports pages with custom body text, headers, footers, and optional encryption.
    """
    doc = fitz.open()

    for idx, page_body in enumerate(pages_text):
        page_num = idx + 1
        page = doc.new_page(width=612, height=792)  # Standard US Letter dimensions

        # Top header if specified (y0 < 10% of height)
        if header_text:
            header = f"{header_text} - Page {page_num}"
            page.insert_text(fitz.Point(50, 30), header, fontsize=10)

        # Body text
        page.insert_text(fitz.Point(50, 100), page_body, fontsize=12)

        # Bottom footer if specified (y1 > 90% of height)
        if footer_text:
            footer = f"{footer_text} | Page {page_num}"
            page.insert_text(fitz.Point(50, 750), footer, fontsize=10)

    if encrypted:
        # Save with encryption permissions
        pdf_bytes = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    else:
        pdf_bytes = doc.tobytes()

    doc.close()
    return pdf_bytes


def create_scanned_test_pdf_bytes() -> bytes:
    """Generate a PDF page with an image/drawing but no extractable text."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Draw a line (visual element) without inserting any text block
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 50), fitz.Point(200, 200))
    shape.finish(color=(1, 0, 0))
    shape.commit()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
