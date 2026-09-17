from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ingestion.loader import PageText, PageTextBlock


@dataclass(frozen=True)
class CleanedPage:
    page_number: int  # 1-based as printed
    text: str


def clean_block_text(text: str) -> str:
    """
    Clean individual block text:
    1. NFKC Unicode normalization.
    2. Soft-hyphen removal.
    3. Conservative dehyphenation (word-\\nlowercase continuation -> joined word).
    4. Normalize internal whitespace within lines.
    """
    if not text:
        return ""

    # 1. Unicode normalization (ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Soft-hyphen removal
    text = text.replace("\u00ad", "")

    # 3. Conservative dehyphenation: only join when line-break hyphen
    # is followed by a lowercase letter continuation.
    text = re.sub(r"(\b[a-zA-Z]+)-\n([a-z]\w*)", r"\1\2", text)

    # 4. Normalize whitespace per line while keeping line breaks intact
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _normalize_furniture_pattern(text: str) -> str:
    """
    Normalize text for furniture frequency detection by masking page numbers/digits.
    e.g. "Page 1 of 10" -> "page <num> of <num>"
    """
    lowered = text.strip().lower()
    return re.sub(r"\b\d+\b", "<num>", lowered)


def detect_page_furniture(pages: list[PageText]) -> set[str]:
    """
    Detect repeated header/footer text patterns using position + frequency signals.
    Top margin: y0 < 10% of page height.
    Bottom margin: y1 > 90% of page height.
    Requirement: Doc must have >= 3 pages, and pattern must appear in top/bottom region
    on >= 2 pages and >= 40% of document pages.
    """
    num_pages = len(pages)
    if num_pages < 3:
        return set()

    candidate_counts: Counter[str] = Counter()

    for page in pages:
        page_candidates: set[str] = set()
        h = page.page_height

        for block in page.blocks:
            is_top = block.y0 < h * 0.10
            is_bottom = block.y1 > h * 0.90

            if is_top or is_bottom:
                cleaned = clean_block_text(block.text)
                if cleaned:
                    norm = _normalize_furniture_pattern(cleaned)
                    page_candidates.add(norm)

        for norm in page_candidates:
            candidate_counts[norm] += 1

    furniture_patterns: set[str] = set()
    min_occurrence = max(2, int(num_pages * 0.4))

    for norm_pattern, count in candidate_counts.items():
        if count >= min_occurrence:
            furniture_patterns.add(norm_pattern)

    return furniture_patterns


def clean_pages(pages: list[PageText]) -> tuple[list[CleanedPage], list[str]]:
    """
    Clean extracted pages:
    - Detect and strip repeated headers/footers.
    - Clean block text (Unicode, dehyphenation, whitespace).
    - Reassemble cleaned blocks per page separated by double line breaks.
    Returns list of CleanedPage objects and list of removed furniture patterns.
    """
    furniture_patterns = detect_page_furniture(pages)
    cleaned_pages: list[CleanedPage] = []
    removed_patterns_log: list[str] = sorted(furniture_patterns)

    for page in pages:
        h = page.page_height
        valid_page_blocks: list[PageTextBlock] = []

        for block in page.blocks:
            is_top = block.y0 < h * 0.10
            is_bottom = block.y1 > h * 0.90

            if is_top or is_bottom:
                cleaned_candidate = clean_block_text(block.text)
                norm = _normalize_furniture_pattern(cleaned_candidate)
                if norm in furniture_patterns:
                    # Skip furniture block
                    continue

            valid_page_blocks.append(block)

        # Process and join remaining blocks
        cleaned_block_texts: list[str] = []
        for block in valid_page_blocks:
            c_text = clean_block_text(block.text)
            if c_text:
                cleaned_block_texts.append(c_text)

        full_cleaned_text = "\n\n".join(cleaned_block_texts)
        cleaned_pages.append(
            CleanedPage(
                page_number=page.page_number,
                text=full_cleaned_text,
            )
        )

    return cleaned_pages, removed_patterns_log
