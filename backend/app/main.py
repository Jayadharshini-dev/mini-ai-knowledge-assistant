from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.config import settings
from backend.app.models import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model pre-loading / initialization hook will go here in later phases
    yield


app = FastAPI(
    title="Mini AI Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    resolved_config = {
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "MAX_UPLOAD_MB": settings.MAX_UPLOAD_MB,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "VECTOR_DIM": settings.VECTOR_DIM,
        "INDEX_TYPE": settings.INDEX_TYPE,
        "INDEX_DIR": str(settings.INDEX_DIR),
        "DOCUMENTS_DIR": str(settings.DOCUMENTS_DIR),
        "TOP_K": settings.TOP_K,
        "RELEVANCE_THRESHOLD": settings.RELEVANCE_THRESHOLD,
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "LLM_MODEL": settings.LLM_MODEL,
        "GEMINI_API_KEY_SET": bool(settings.GEMINI_API_KEY),
    }
    return HealthResponse(status="ok", config=resolved_config)
