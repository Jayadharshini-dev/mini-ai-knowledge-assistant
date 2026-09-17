from __future__ import annotations

from dataclasses import dataclass

import fitz
from fastapi import status

from backend.app.errors import AppException, ErrorCode


@dataclass(frozen=True)
class PageTextBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based as printed
    blocks: list[PageTextBlock]
    full_text: str
    page_width: float
    page_height: float


def load_pdf_pages(file_bytes: bytes, filename: str) -> list[PageText]:
    """
    Extract page-aware text blocks from a PDF using PyMuPDF (fitz).
    Converts 0-based index to 1-based page numbers.
    Validates encryption, parseability, and extractable text content.
    """
    if not file_bytes:
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="PDF file buffer is empty",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message=f"Failed to parse PDF document '{filename}': {exc}",
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc

    if doc.is_encrypted:
        doc.close()
        raise AppException(
            code=ErrorCode.ENCRYPTED_PDF,
            message=f"PDF document '{filename}' is encrypted or password-protected",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    pages: list[PageText] = []
    total_text_length = 0

    try:
        for idx, page in enumerate(doc):
            page_number = idx + 1  # 1-based page number
            rect = page.rect
            page_width = float(rect.width)
            page_height = float(rect.height)

            # Blocks: (x0, y0, x1, y1, text, block_no, block_type)
            raw_blocks = page.get_text("blocks", sort=True)
            text_blocks: list[PageTextBlock] = []
            page_text_parts: list[str] = []

            for b in raw_blocks:
                # block_type == 0 indicates text block (block_type == 1 is image)
                if len(b) >= 7 and b[6] == 0:
                    x0, y0, x1, y1, block_text, block_no = (
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                        str(b[4]),
                        int(b[5]),
                    )
                    cleaned_btext = block_text.strip()
                    if cleaned_btext:
                        text_blocks.append(
                            PageTextBlock(
                                x0=x0,
                                y0=y0,
                                x1=x1,
                                y1=y1,
                                text=block_text,
                                block_no=block_no,
                            )
                        )
                        page_text_parts.append(cleaned_btext)

            full_page_text = "\n\n".join(page_text_parts)
            total_text_length += len(full_page_text.strip())

            pages.append(
                PageText(
                    page_number=page_number,
                    blocks=text_blocks,
                    full_text=full_page_text,
                    page_width=page_width,
                    page_height=page_height,
                )
            )
    finally:
        doc.close()

    if total_text_length == 0:
        msg = f"No extractable text found in '{filename}'. Scanned PDFs require OCR."
        raise AppException(
            code=ErrorCode.NO_EXTRACTABLE_TEXT,
            message=msg,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    return pages
