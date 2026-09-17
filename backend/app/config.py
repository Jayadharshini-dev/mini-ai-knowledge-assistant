from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_UPLOAD_MB: int = 15

    # Embeddings & Vector Store
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    VECTOR_DIM: int = 384
    INDEX_TYPE: str = "IndexFlatIP"
    INDEX_DIR: Path = Path("data/index")
    DOCUMENTS_DIR: Path = Path("data/documents")

    # Retrieval & Gate
    TOP_K: int = 4
    RELEVANCE_THRESHOLD: float = 0.32

    # LLM Provider
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: Optional[str] = None


# Single shared configuration instance
settings = AppConfig()
