#!/usr/bin/env python3
"""
scripts/inspect_chunks.py
Dump readable chunks and ingestion statistics for a PDF document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from ingestion.chunker import bge_token_counter, chunk_cleaned_pages
from ingestion.cleaner import clean_pages
from ingestion.loader import load_pdf_pages


def inspect_pdf(pdf_path: str, sample_size: int = 10) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: File not found: '{pdf_path}'")
        sys.exit(1)

    file_bytes = path.read_bytes()
    filename = path.name
    size_kb = len(file_bytes) / 1024.0

    print("=" * 80)
    print(f"DOCUMENT INGESTION INSPECTION: {filename}")
    print(f"Path: {path.resolve()}")
    print(f"File Size: {size_kb:.1f} KB")
    print("=" * 80)

    # 1. Extraction
    pages = load_pdf_pages(file_bytes, filename)
    num_pages = len(pages)
    print("\n[1] EXTRACTION")
    print(f"  - Extracted Pages: {num_pages}")
    total_chars = sum(len(p.full_text) for p in pages)
    print(f"  - Total Extracted Characters: {total_chars}")

    # 2. Cleaning
    cleaned_pages, removed_furniture = clean_pages(pages)
    print("\n[2] CLEANING & FURNITURE REMOVAL")
    print(f"  - Cleaned Pages: {len(cleaned_pages)}")
    if removed_furniture:
        print(f"  - Removed Header/Footer Patterns ({len(removed_furniture)}):")
        for pattern in removed_furniture:
            print(f"      * '{pattern}'")
    else:
        print("  - Removed Header/Footer Patterns: None detected")

    # 3. Chunking
    chunks = chunk_cleaned_pages(
        pages=cleaned_pages,
        filename=filename,
        file_bytes=file_bytes,
        token_counter=bge_token_counter,
    )

    c_size = settings.CHUNK_SIZE
    c_overlap = settings.CHUNK_OVERLAP
    print(f"\n[3] CHUNKING (Config: CHUNK_SIZE={c_size}, CHUNK_OVERLAP={c_overlap})")
    print(f"  - Total Chunks Generated: {len(chunks)}")

    if chunks:
        tokens_list = [c.token_estimate for c in chunks]
        min_tok = min(tokens_list)
        max_tok = max(tokens_list)
        avg_tok = sum(tokens_list) / len(tokens_list)
        print(f"  - Token Stats: Min={min_tok}, Max={max_tok}, Avg={avg_tok:.1f}")
        passed_limit = "PASSED" if max_tok <= 510 else "FAILED"
        print(f"  - BGE Token Limit (<= 510) Check: {passed_limit}")

    # 4. Sample Chunks Display
    num_sample = min(sample_size, len(chunks))
    print(f"\n[4] REPRESENTATIVE CHUNK SAMPLE ({num_sample} of {len(chunks)})")
    print("-" * 80)

    for i, chunk in enumerate(chunks[:num_sample]):
        print(f"Chunk #{i + 1} | ID: {chunk.chunk_id}")
        details = (
            f"Page: {chunk.page} | Index: {chunk.chunk_index} | "
            f"Tokens: {chunk.token_estimate} | Chars: {chunk.char_count}"
        )
        print(details)
        print("Text Excerpt:")
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f'  "{preview}"')
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PDF extraction, cleaning, and chunking."
    )
    parser.add_argument("pdf_path", type=str, help="Path to PDF document")
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample chunks to display (default: 10)",
    )
    args = parser.parse_args()

    inspect_pdf(args.pdf_path, args.sample)


if __name__ == "__main__":
    main()
