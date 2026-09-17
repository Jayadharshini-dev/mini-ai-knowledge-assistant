from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer

from backend.app.config import settings
from backend.app.models import Chunk
from ingestion.cleaner import CleanedPage

_CACHED_TOKENIZER = None


def get_production_tokenizer():
    """Load and cache Hugging Face BGE tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
    return _CACHED_TOKENIZER


def bge_token_counter(text: str) -> int:
    """Production token counter using BGE AutoTokenizer without special tokens."""
    if not text:
        return 0
    tokenizer = get_production_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def default_word_counter(text: str) -> int:
    """Fast lightweight token counter fallback for fast network-free unit tests."""
    if not text:
        return 0
    return len(text.split())


@dataclass(frozen=True)
class Unit:
    text: str
    tokens: int


def _split_text_to_units(
    text: str,
    token_counter: Callable[[str], int],
    max_chunk_tokens: int,
) -> list[Unit]:
    """
    Escalating fallback algorithm:
    Paragraph -> Sentence -> Word -> Pathological Token
    Returns a list of units, each guaranteed to have tokens <= max_chunk_tokens.
    """
    units: list[Unit] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for para in paragraphs:
        para_tokens = token_counter(para)
        if para_tokens <= max_chunk_tokens:
            units.append(Unit(text=para, tokens=para_tokens))
            continue

        # Paragraph > max_chunk_tokens: split into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        for sent in sentences:
            sent_tokens = token_counter(sent)
            if sent_tokens <= max_chunk_tokens:
                units.append(Unit(text=sent, tokens=sent_tokens))
                continue

            # Sentence > max_chunk_tokens: split into words
            words = [w for w in sent.split() if w]
            for word in words:
                word_tokens = token_counter(word)
                if word_tokens <= max_chunk_tokens:
                    units.append(Unit(text=word, tokens=word_tokens))
                    continue

                # Pathological token > max_chunk_tokens: hard character slice
                slice_len = max(1, len(word) // (word_tokens // max_chunk_tokens + 1))
                for i in range(0, len(word), slice_len):
                    part = word[i : i + slice_len]
                    part_tokens = token_counter(part)
                    units.append(Unit(text=part, tokens=part_tokens))

    return units


def _generate_doc_id(file_bytes: bytes) -> str:
    """Generate 12-character hex SHA-256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()[:12]


def _clean_doc_prefix(filename: str) -> str:
    """Clean filename into a safe ID prefix."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return cleaned or "doc"


def chunk_cleaned_pages(
    pages: list[CleanedPage],
    filename: str,
    file_bytes: bytes,
    token_counter: Callable[[str], int] = bge_token_counter,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Turn cleaned pages into token-aware, metadata-carrying Chunk objects.
    Enforces page-aware chunking (chunks & overlap NEVER cross page boundaries).
    """
    target_size = chunk_size or settings.CHUNK_SIZE
    target_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    max_limit = min(target_size, 510)  # Hard invariant limit

    doc_id = _generate_doc_id(file_bytes)
    doc_prefix = _clean_doc_prefix(filename)

    chunks: list[Chunk] = []
    global_chunk_index = 1

    for page in pages:
        if not page.text.strip():
            continue

        page_units = _split_text_to_units(
            text=page.text,
            token_counter=token_counter,
            max_chunk_tokens=max_limit,
        )

        if not page_units:
            continue

        unit_idx = 0
        num_units = len(page_units)

        while unit_idx < num_units:
            current_units: list[Unit] = []
            current_tokens = 0

            # Accumulate units for the current chunk
            while unit_idx < num_units:
                unit = page_units[unit_idx]
                if current_units and (current_tokens + unit.tokens > max_limit):
                    break
                current_units.append(unit)
                current_tokens += unit.tokens
                unit_idx += 1

            if not current_units:
                break

            # Join unit text cleanly
            chunk_text = "\n\n".join(u.text for u in current_units)
            measured_tokens = token_counter(chunk_text)

            c_num = page.page_number
            chunk_id = f"{doc_prefix}__p{c_num:03d}__c{global_chunk_index:04d}"

            chunk_obj = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                document=filename,
                page=page.page_number,
                chunk_index=global_chunk_index,
                text=chunk_text,
                char_count=len(chunk_text),
                token_estimate=measured_tokens,
            )
            chunks.append(chunk_obj)
            global_chunk_index += 1

            # If all units on this page processed, end loop for this page
            if unit_idx >= num_units:
                break

            # Calculate overlap from whole trailing units for next chunk on SAME page
            overlap_units: list[Unit] = []
            overlap_tokens = 0
            max_allowed_overlap_tokens = min(target_overlap, current_tokens // 2)

            for u in reversed(current_units):
                if overlap_tokens + u.tokens > max_allowed_overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_tokens += u.tokens

            # Ensure we consume at least one new unit in the next chunk
            units_consumed = len(current_units) - len(overlap_units)
            if units_consumed <= 0 and len(current_units) > 1:
                # Force at least 1 unit to be consumed
                overlap_units = current_units[1:]

            # Rewind unit_idx back by the number of overlap units carried forward
            unit_idx = unit_idx - len(overlap_units)

    return chunks
