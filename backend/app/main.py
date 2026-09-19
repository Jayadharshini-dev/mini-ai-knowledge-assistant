import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.deps import (
    get_document_registry,
    get_embedder,
    get_vector_store,
)
from backend.app.errors import AppException, ErrorCode
from backend.app.models import HealthResponse
from backend.app.routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm VectorStore, DocumentRegistry, and Embedder on app startup
    try:
        get_vector_store()
        get_document_registry()
        get_embedder()
    except Exception:
        logger.exception("Failed to pre-warm application state during lifespan startup")
    yield


app = FastAPI(
    title="Mini AI Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code.value,
            "message": exc.message,
            "retryable": exc.retryable,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": ErrorCode.INTERNAL_ERROR.value,
            "message": "An internal server error occurred.",
            "retryable": False,
        },
    )


app.include_router(api_router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    resolved_config = {
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "MAX_UPLOAD_MB": settings.MAX_UPLOAD_MB,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "VECTOR_DIM": settings.VECTOR_DIM,
        "INDEX_TYPE": settings.INDEX_TYPE,
        "TOP_K": settings.TOP_K,
        "RELEVANCE_THRESHOLD": settings.RELEVANCE_THRESHOLD,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_MODEL": settings.LLM_MODEL,
        "GEMINI_API_KEY_SET": bool(settings.GEMINI_API_KEY),
    }
    return HealthResponse(status="ok", config=resolved_config)
