from __future__ import annotations

import re

from pydantic import BaseModel, Field

from backend.app.models import Citation, RetrievedChunk


class CitationValidationResult(BaseModel):
    """
    Result of validating model-generated citation markers against supplied context
    chunks.
    """

    clean_text: str
    valid_citations: list[Citation] = Field(default_factory=list)
    citations_emitted: int = 0
    citations_dropped: int = 0
    invalid_citation_ids: list[str] = Field(default_factory=list)


def validate_citations(
    raw_text: str, supplied_chunks: list[RetrievedChunk]
) -> CitationValidationResult:
    """
    Validate all citation markers present in raw_text against the supplied retrieved
    chunks.

    Invariant: A citation is valid ONLY if its chunk_id exists in supplied_chunks.
    Invalid citation markers are stripped from clean_text and counted in
    citations_dropped.
    """
    if not raw_text:
        return CitationValidationResult(clean_text="", valid_citations=[])

    # Map supplied chunk_ids to chunk objects
    chunk_map = {c.chunk.chunk_id: c.chunk for c in supplied_chunks}

    # Find bracketed markers like [chunk_id] or [citation: chunk_id]
    pattern = re.compile(r"\[(?:citation:\s*)?([a-zA-Z0-9_\-\.]+?)\]")

    valid_citations: list[Citation] = []
    seen_valid_ids: set[str] = set()
    invalid_citation_ids: list[str] = []
    citations_emitted_count = 0
    citations_dropped_count = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal citations_emitted_count, citations_dropped_count
        citations_emitted_count += 1
        full_marker = match.group(0)
        clean_id = match.group(1).strip()

        if clean_id in chunk_map:
            if clean_id not in seen_valid_ids:
                seen_valid_ids.add(clean_id)
                chunk_obj = chunk_map[clean_id]
                valid_citations.append(
                    Citation(
                        n=len(valid_citations) + 1,
                        chunk_id=chunk_obj.chunk_id,
                        document=chunk_obj.document,
                        page=chunk_obj.page,
                    )
                )
            return full_marker
        else:
            citations_dropped_count += 1
            invalid_citation_ids.append(clean_id)
            return ""

    clean_text = pattern.sub(replace_match, raw_text)
    clean_text = re.sub(r" +", " ", clean_text)
    clean_text = re.sub(r" \.", ".", clean_text)
    clean_text = clean_text.strip()

    return CitationValidationResult(
        clean_text=clean_text,
        valid_citations=valid_citations,
        citations_emitted=citations_emitted_count,
        citations_dropped=citations_dropped_count,
        invalid_citation_ids=invalid_citation_ids,
    )
