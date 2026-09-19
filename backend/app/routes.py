from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import StreamingResponse

from backend.app.config import settings
from backend.app.deps import (
    get_document_registry,
    get_embedder,
    get_kb_status,
    get_pipeline,
    get_vector_store,
)
from backend.app.errors import AppException, ErrorCode
from backend.app.models import (
    ChatRequest,
    KnowledgeBaseStatus,
)
from backend.rag.events import EventEmitter, EventType
from backend.rag.outcome import collect_run
from ingestion.chunker import _generate_doc_id, chunk_cleaned_pages
from ingestion.cleaner import clean_pages
from ingestion.loader import load_pdf_pages

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/kb/status", response_model=KnowledgeBaseStatus)
@router.get("/api/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def knowledge_base_status():
    return get_kb_status()


@router.get("/api/documents")
async def list_documents():
    registry = get_document_registry()
    docs = registry.list_documents()
    return {"documents": [d.model_dump() for d in docs]}


@router.post("/api/documents")
async def ingest_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.strip():
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="No file uploaded or filename missing.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message=f"Uploaded file '{filename}' is empty.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        msg = f"File '{filename}' exceeds MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB} MB)"
        raise AppException(
            code=ErrorCode.FILE_TOO_LARGE,
            message=msg,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    async def sse_ingest_generator():
        emitter = EventEmitter()
        try:
            doc_id = _generate_doc_id(file_bytes)
            registry = get_document_registry()

            # 1. Duplicate check
            if registry.is_duplicate(doc_id):
                dup_evt = emitter.emit(
                    EventType.DOCUMENT_DUPLICATE,
                    "warn",
                    "Document hash already indexed",
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "message": "Document content hash already exists in registry.",
                    },
                )
                yield f"event: done\ndata: {json.dumps(dup_evt.to_dict())}\n\n"
                return

            # 2. DOCUMENT_RECEIVED
            rec_evt = emitter.emit(
                EventType.DOCUMENT_RECEIVED,
                "ok",
                "Document upload received",
                {
                    "filename": filename,
                    "size_bytes": len(file_bytes),
                    "doc_id": doc_id,
                },
            )
            yield f"event: trace\ndata: {json.dumps(rec_evt.to_dict())}\n\n"

            # 3. TEXT_EXTRACTED
            pages = await asyncio.to_thread(load_pdf_pages, file_bytes, filename)
            extracted_chars = sum(len(p.full_text) for p in pages)
            txt_evt = emitter.emit(
                EventType.TEXT_EXTRACTED,
                "ok",
                f"Extracted {len(pages)} pages",
                {
                    "pages": len(pages),
                    "extracted_chars": extracted_chars,
                },
            )
            yield f"event: trace\ndata: {json.dumps(txt_evt.to_dict())}\n\n"

            # 4. TEXT_CLEANED
            cleaned_pages, furniture = await asyncio.to_thread(clean_pages, pages)
            cln_evt = emitter.emit(
                EventType.TEXT_CLEANED,
                "ok",
                f"Cleaned {len(cleaned_pages)} pages",
                {
                    "furniture_patterns_removed": len(furniture),
                    "cleaned_pages": len(cleaned_pages),
                },
            )
            yield f"event: trace\ndata: {json.dumps(cln_evt.to_dict())}\n\n"

            # 5. CHUNKED
            chunks = await asyncio.to_thread(
                chunk_cleaned_pages, cleaned_pages, filename, file_bytes
            )
            avg_tokens = (
                int(sum(c.token_estimate for c in chunks) / len(chunks))
                if chunks
                else 0
            )
            chk_evt = emitter.emit(
                EventType.CHUNKED,
                "ok",
                f"Created {len(chunks)} chunks",
                {
                    "chunks_created": len(chunks),
                    "avg_chunk_tokens": avg_tokens,
                },
            )
            yield f"event: trace\ndata: {json.dumps(chk_evt.to_dict())}\n\n"

            # 6. EMBEDDED
            embedder = get_embedder()
            chunk_texts = [c.text for c in chunks]
            vectors = await asyncio.to_thread(embedder.embed_chunks, chunk_texts)
            emb_evt = emitter.emit(
                EventType.EMBEDDED,
                "ok",
                f"Embedded {len(vectors)} chunk vectors",
                {
                    "vectors_computed": len(vectors),
                    "dim": embedder.dimension,
                },
            )
            yield f"event: trace\ndata: {json.dumps(emb_evt.to_dict())}\n\n"

            # 7. INDEXED
            store = get_vector_store()
            store.add_chunks(chunks, vectors)
            await asyncio.to_thread(store.save_index, settings.INDEX_DIR)
            idx_evt = emitter.emit(
                EventType.INDEXED,
                "ok",
                f"Indexed {len(chunks)} chunks into FAISS",
                {
                    "total_vectors": store.num_vectors,
                    "index_type": store.index_type,
                },
            )
            yield f"event: trace\ndata: {json.dumps(idx_evt.to_dict())}\n\n"

            # 8. COMPLETE
            record = registry.register(
                doc_id=doc_id,
                filename=filename,
                pages=len(pages),
                chunks=len(chunks),
            )
            comp_evt = emitter.emit(
                EventType.COMPLETE,
                "ok",
                "Document ingestion complete",
                record.model_dump(),
            )
            yield f"event: done\ndata: {json.dumps(comp_evt.to_dict())}\n\n"

        except AppException as exc:
            err_evt = emitter.emit(
                EventType.ERROR,
                "error",
                "Ingestion failed",
                {
                    "code": exc.code.value,
                    "message": exc.message,
                    "retryable": exc.retryable,
                },
            )
            yield f"event: done\ndata: {json.dumps(err_evt.to_dict())}\n\n"
        except Exception:
            logger.exception("Ingestion failed due to unhandled error")
            err_evt = emitter.emit(
                EventType.ERROR,
                "error",
                "Ingestion failed",
                {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": "An internal error occurred during document ingestion.",
                    "retryable": False,
                },
            )
            yield f"event: done\ndata: {json.dumps(err_evt.to_dict())}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_ingest_generator(), media_type="text/event-stream", headers=headers
    )


@router.post("/api/chat")
async def chat_json(request: ChatRequest):
    cleaned_question = request.question.strip() if request.question else ""
    if not cleaned_question:
        raise AppException(
            code=ErrorCode.EMPTY_QUESTION,
            message="Question cannot be empty or whitespace.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    store = get_vector_store()
    if store.num_vectors == 0:
        raise AppException(
            code=ErrorCode.NO_DOCUMENTS,
            message="Knowledge base has no indexed documents.",
            status_code=status.HTTP_409_CONFLICT,
        )

    pipeline = get_pipeline()
    collected = await collect_run(pipeline.run(request.question))
    events = collected.events

    terminal_event = events[-1]
    if terminal_event.type == EventType.ERROR:
        err_code_str = terminal_event.detail.get("code", ErrorCode.INTERNAL_ERROR.value)
        err_msg = terminal_event.detail.get("message", "Pipeline error occurred.")
        try:
            err_code = ErrorCode(err_code_str)
        except ValueError:
            err_code = ErrorCode.INTERNAL_ERROR

        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if err_code in (ErrorCode.PROVIDER_UNAVAILABLE, ErrorCode.PROVIDER_AUTH):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        raise AppException(
            code=err_code,
            message=err_msg,
            status_code=status_code,
            retryable=terminal_event.detail.get("retryable", False),
        )

    trace = [e.to_dict() for e in events]
    return {
        **terminal_event.detail,
        "trace": trace,
    }


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    cleaned_question = request.question.strip() if request.question else ""
    if not cleaned_question:
        raise AppException(
            code=ErrorCode.EMPTY_QUESTION,
            message="Question cannot be empty or whitespace.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    store = get_vector_store()
    if store.num_vectors == 0:
        raise AppException(
            code=ErrorCode.NO_DOCUMENTS,
            message="Knowledge base has no indexed documents.",
            status_code=status.HTTP_409_CONFLICT,
        )

    pipeline = get_pipeline()

    async def sse_generator():
        try:
            async for event in pipeline.run(request.question):
                event_name = (
                    "done"
                    if event.type in (EventType.COMPLETE, EventType.ERROR)
                    else "trace"
                )
                data_str = json.dumps(event.to_dict(), ensure_ascii=False)
                yield f"event: {event_name}\ndata: {data_str}\n\n"
        except Exception:
            logger.exception("Error during SSE stream execution")
            err_detail = {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "An internal server error occurred during streaming.",
                "retryable": False,
            }
            yield f"event: error\ndata: {json.dumps(err_detail)}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_generator(), media_type="text/event-stream", headers=headers
    )
