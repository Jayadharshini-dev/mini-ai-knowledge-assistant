from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.models import DocumentRecord


class DocumentRegistry:
    """
    JSON-backed document registry for duplicate protection and tracking.
    Tracks ingested documents by SHA-256 content hash (doc_id).
    """

    def __init__(self, registry_dir: Path | str | None = None):
        self.dir = Path(registry_dir or settings.INDEX_DIR)
        self.registry_path = self.dir / "registry.json"
        self.documents: dict[str, DocumentRecord] = {}
        self.load()

    def load(self) -> None:
        """Load registry records from registry.json if present."""
        if not self.registry_path.exists():
            self.documents = {}
            return

        try:
            with open(self.registry_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            self.documents = {
                doc_id: DocumentRecord(**rec) for doc_id, rec in raw_data.items()
            }
        except Exception:
            self.documents = {}

    def save(self) -> None:
        """Atomically save registry records to registry.json."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".json.tmp")
        raw_dict: dict[str, Any] = {
            doc_id: doc.model_dump() for doc_id, doc in self.documents.items()
        }
        json_bytes = json.dumps(raw_dict, indent=2, ensure_ascii=False).encode("utf-8")

        with open(tmp_path, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.registry_path)

    def is_duplicate(self, doc_id: str) -> bool:
        """Return True if doc_id has already been ingested into the registry."""
        return doc_id in self.documents

    def register(
        self,
        doc_id: str,
        filename: str,
        pages: int,
        chunks: int,
        status: str = "indexed",
    ) -> DocumentRecord:
        """Register a new document record and persist changes."""
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
            chunks=chunks,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            status=status,
        )
        self.documents[doc_id] = record
        self.save()
        return record

    def list_documents(self) -> list[DocumentRecord]:
        """Return list of registered document records sorted by filename."""
        return sorted(self.documents.values(), key=lambda d: d.filename)
