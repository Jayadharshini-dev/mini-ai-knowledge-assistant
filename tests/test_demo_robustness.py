from __future__ import annotations

from collections.abc import Generator

import fitz
import pytest
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.deps import reset_dependencies
from backend.app.main import app


@pytest.fixture(autouse=True)
def cleanup_deps() -> Generator[None, None, None]:
    reset_dependencies()
    yield
    reset_dependencies()


def test_upload_empty_file_returns_400_invalid_pdf():
    client = TestClient(app)
    response = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "INVALID_PDF"
    assert "empty" in data["message"].lower()
    assert "Traceback" not in data["message"]
    assert "Exception" not in data["message"]


def test_upload_invalid_pdf_bytes_returns_400_invalid_pdf():
    client = TestClient(app)
    response = client.post(
        "/api/documents",
        files={"file": ("corrupted.pdf", b"NOT_A_REAL_PDF_BUFFER", "application/pdf")},
    )
    # The endpoint streams SSE events for ingestion failures
    assert response.status_code == 200
    events = response.text.strip().split("\n\n")
    done_event = [e for e in events if "event: done" in e or "event: error" in e]
    assert len(done_event) > 0
    assert "INVALID_PDF" in done_event[0]
    assert "Traceback" not in done_event[0]


def test_upload_encrypted_pdf_returns_400_encrypted_pdf():
    # Construct an encrypted PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Encrypted content")
    encrypted_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="user123", owner_pw="owner123"
    )
    doc.close()

    client = TestClient(app)
    response = client.post(
        "/api/documents",
        files={"file": ("encrypted.pdf", encrypted_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    assert "ENCRYPTED_PDF" in response.text
    assert "encrypted" in response.text.lower()
    assert "Traceback" not in response.text


def test_upload_scanned_pdf_no_text_returns_422_no_extractable_text():
    # Construct a PDF page with no text blocks
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(10, 10, 100, 100), fill=(0.8, 0.8, 0.8))
    no_text_bytes = doc.tobytes()
    doc.close()

    client = TestClient(app)
    response = client.post(
        "/api/documents",
        files={"file": ("scanned.pdf", no_text_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    assert "NO_EXTRACTABLE_TEXT" in response.text
    assert "No extractable text" in response.text
    assert "Traceback" not in response.text


def test_upload_oversized_file_returns_413_file_too_large():
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    oversized_bytes = b"X" * (max_bytes + 1024)

    client = TestClient(app)
    response = client.post(
        "/api/documents",
        files={"file": ("oversized.pdf", oversized_bytes, "application/pdf")},
    )
    assert response.status_code == 413
    data = response.json()
    assert data["code"] == "FILE_TOO_LARGE"
    assert f"MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB} MB)" in data["message"]
    assert "Traceback" not in data["message"]


def test_chat_empty_question_returns_400_empty_question():
    client = TestClient(app)
    response = client.post("/api/chat", json={"question": "   "})
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "EMPTY_QUESTION"
    assert "empty" in data["message"].lower()
    assert "Traceback" not in data["message"]


def test_chat_stream_empty_question_returns_400_empty_question():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "   "})
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == "EMPTY_QUESTION"
    assert "Traceback" not in data["message"]
