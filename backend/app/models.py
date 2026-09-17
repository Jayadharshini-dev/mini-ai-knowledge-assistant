from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Query stream
    QUERY_RECEIVED = "QUERY_RECEIVED"
    EMBEDDING_STARTED = "EMBEDDING_STARTED"
    EMBEDDING_COMPLETED = "EMBEDDING_COMPLETED"
    RETRIEVAL_STARTED = "RETRIEVAL_STARTED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    EVIDENCE_SELECTED = "EVIDENCE_SELECTED"
    ABSTAINED = "ABSTAINED"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    GENERATION_STARTED = "GENERATION_STARTED"
    GENERATION_COMPLETED = "GENERATION_COMPLETED"
    GENERATION_SKIPPED = "GENERATION_SKIPPED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

    # Ingestion stream
    DOCUMENT_RECEIVED = "DOCUMENT_RECEIVED"
    DOCUMENT_DUPLICATE = "DOCUMENT_DUPLICATE"
    TEXT_EXTRACTED = "TEXT_EXTRACTED"
    TEXT_CLEANED = "TEXT_CLEANED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    document: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    above_threshold: bool


class Citation(BaseModel):
    n: int
    chunk_id: str
    document: str
    page: int


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    indexed_at: str
    status: Literal["indexed", "failed"]


class TraceEvent(BaseModel):
    seq: int
    type: EventType
    status: Literal["active", "ok", "warn", "error"]
    t_ms: int
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseStatus(BaseModel):
    state: Literal["empty", "indexing", "ready", "error"]
    documents: int
    chunks: int
    vector_dim: int
    index_type: str
    index_size_bytes: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_available: bool
    top_k: int
    relevance_threshold: float
    retriever: Literal["dense", "hybrid"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    config: dict[str, Any]
