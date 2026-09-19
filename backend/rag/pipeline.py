from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Callable

from backend.app.errors import AppException, ErrorCode
from backend.generation.base import GenerationContext, LLMProvider, RawGenerationResult
from backend.generation.citation_validator import CitationValidationResult
from backend.rag.events import EventEmitter, EventType, TraceEvent
from backend.rag.failures import map_exception_to_error_detail
from retrieval.relevance import RelevanceDecision


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy scalars, tuples, and Pydantic models into pure Python
    JSON-serializable primitives.
    Strictly raises TypeError on unknown unsupported types.
    """
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    elif hasattr(obj, "item") and callable(obj.item):
        return _to_json_safe(obj.item())
    elif hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _to_json_safe(obj.model_dump())
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class RagPipeline:
    """
    RAG Pipeline orchestrator executing dense retrieval, relevance evaluation,
    context building, LLM generation, and citation validation trace stream.
    """

    def __init__(
        self,
        *,
        retriever: Any,
        relevance: Callable[[list[Any]], RelevanceDecision],
        context_builder: Callable[..., GenerationContext],
        provider: LLMProvider,
        citation_validator: Callable[..., CitationValidationResult],
    ) -> None:
        self.retriever = retriever
        self.relevance = relevance
        self.context_builder = context_builder
        self.provider = provider
        self.citation_validator = citation_validator

    async def run(self, question: str) -> AsyncIterator[TraceEvent]:
        """
        Execute the RAG pipeline as an async generator emitting TraceEvents.

        Invariant: Every run ends with exactly ONE terminal event (COMPLETE or ERROR).
        All blocking calls use asyncio.to_thread.
        """
        emitter = EventEmitter()
        current_stage = "query"

        try:
            # 1. QUERY_RECEIVED
            searched_vectors = self.retriever.store.num_vectors
            yield emitter.emit(
                EventType.QUERY_RECEIVED,
                "ok",
                "Query received",
                _to_json_safe(
                    {
                        "question": question,
                        "index_size": searched_vectors,
                    }
                ),
            )

            # 2. RETRIEVAL_STARTED
            current_stage = "retrieve"
            yield emitter.emit(
                EventType.RETRIEVAL_STARTED, "active", "Searching index", {}
            )

            # Execute retrieval via asyncio.to_thread
            t_search_start = time.perf_counter()
            retrieved = await asyncio.to_thread(self.retriever.retrieve, question)
            search_ms = int((time.perf_counter() - t_search_start) * 1000)

            # 3. RETRIEVAL_COMPLETED
            chunks_json = [_to_json_safe(c.model_dump()) for c in retrieved]
            index_type = self.retriever.store.index_type
            yield emitter.emit(
                EventType.RETRIEVAL_COMPLETED,
                "ok",
                f"Retrieved {len(retrieved)} candidates",
                _to_json_safe(
                    {
                        "searched_vectors": searched_vectors,
                        "index_type": index_type,
                        "returned": len(retrieved),
                        "chunks": chunks_json,
                    }
                ),
            )

            # 4. EVIDENCE_SELECTED
            current_stage = "relevance"
            decision = self.relevance(retrieved)
            ev_status = "ok" if decision.decision == "proceed" else "warn"
            yield emitter.emit(
                EventType.EVIDENCE_SELECTED,
                ev_status,
                f"Selected {len(decision.selected_chunks)} chunk(s)",
                _to_json_safe(
                    {
                        "threshold": decision.threshold,
                        "metric": "cosine",
                        "top_score": decision.top_score,
                        "selected": len(decision.selected_chunks),
                        "rejected": len(decision.rejected_chunks),
                        "decision": decision.decision,
                    }
                ),
            )

            # 5. ABSTENTION CHECK
            if decision.decision == "abstain":
                yield emitter.emit(
                    EventType.ABSTAINED,
                    "warn",
                    "No passage above relevance threshold",
                    _to_json_safe(
                        {
                            "reason": decision.reason or "no_passage_above_threshold",
                            "threshold": decision.threshold,
                            "top_score": decision.top_score,
                            "message": decision.message,
                            "chunks": chunks_json,
                        }
                    ),
                )

                # Terminal COMPLETE event for abstention (evidence preserved)
                yield emitter.emit(
                    EventType.COMPLETE,
                    "ok",
                    "Run completed (abstained)",
                    _to_json_safe(
                        {
                            "answer": None,
                            "citations": [],
                            "chunks": chunks_json,
                            "elapsed_ms": emitter.t_ms,
                            "abstained": True,
                            "degraded": None,
                            "timings": {
                                "search_ms": search_ms,
                                "context_ms": 0,
                                "generate_ms": 0,
                            },
                        }
                    ),
                )
                return

            # 6. CONTEXT_BUILT
            current_stage = "context"
            t_ctx_start = time.perf_counter()
            context = self.context_builder(question, decision.selected_chunks)
            context_ms = int((time.perf_counter() - t_ctx_start) * 1000)

            yield emitter.emit(
                EventType.CONTEXT_BUILT,
                "ok",
                "Context assembled",
                _to_json_safe(
                    {
                        "passages": context.passages_count,
                        "context_chars": context.context_chars,
                        "token_estimate": context.token_estimate,
                        "truncated": context.truncated,
                    }
                ),
            )

            # 7. PROVIDER CHECK & GENERATION
            current_stage = "generate"
            if not self.provider.is_available():
                yield emitter.emit(
                    EventType.GENERATION_SKIPPED,
                    "warn",
                    "No provider configured — evidence-only mode",
                    _to_json_safe(
                        {"reason": "no_provider_configured", "mode": "evidence_only"}
                    ),
                )

                yield emitter.emit(
                    EventType.COMPLETE,
                    "ok",
                    "Run completed (evidence only)",
                    _to_json_safe(
                        {
                            "answer": None,
                            "citations": [],
                            "chunks": chunks_json,
                            "elapsed_ms": emitter.t_ms,
                            "abstained": False,
                            "degraded": "no_provider",
                            "timings": {
                                "search_ms": search_ms,
                                "context_ms": context_ms,
                                "generate_ms": 0,
                            },
                        }
                    ),
                )
                return

            # Provider is available
            yield emitter.emit(
                EventType.GENERATION_STARTED,
                "active",
                f"Calling {self.provider.provider_name}",
                _to_json_safe(
                    {
                        "provider": self.provider.provider_name,
                        "model": self.provider.model_name,
                    }
                ),
            )

            # Call provider.generate via asyncio.to_thread
            raw_res: RawGenerationResult = await asyncio.to_thread(
                self.provider.generate, context
            )

            if raw_res.status != "completed":
                # Auth failures are configuration errors -> terminate stream as ERROR
                if raw_res.error_code == ErrorCode.PROVIDER_AUTH:
                    raise AppException(
                        code=ErrorCode.PROVIDER_AUTH,
                        message=raw_res.error_message
                        or "Provider authentication failed.",
                    )

                # Transient provider failures -> degraded path
                degraded_reason = (
                    raw_res.error_code.value if raw_res.error_code else raw_res.status
                )
                yield emitter.emit(
                    EventType.GENERATION_SKIPPED,
                    "warn",
                    f"Provider execution degraded ({raw_res.status})",
                    _to_json_safe(
                        {
                            "reason": degraded_reason,
                            "status": raw_res.status,
                            "mode": "evidence_only",
                        }
                    ),
                )

                yield emitter.emit(
                    EventType.COMPLETE,
                    "ok",
                    "Run completed (provider degraded)",
                    _to_json_safe(
                        {
                            "answer": None,
                            "citations": [],
                            "chunks": chunks_json,
                            "elapsed_ms": emitter.t_ms,
                            "abstained": False,
                            "degraded": degraded_reason,
                            "timings": {
                                "search_ms": search_ms,
                                "context_ms": context_ms,
                                "generate_ms": raw_res.elapsed_ms,
                            },
                        }
                    ),
                )
                return

            # Successful provider execution -> Validate citations
            val_res = self.citation_validator(
                raw_res.text or "", decision.selected_chunks
            )

            yield emitter.emit(
                EventType.GENERATION_COMPLETED,
                "ok",
                "Generation completed",
                _to_json_safe(
                    {
                        "citations_emitted": val_res.citations_emitted,
                        "citations_dropped": val_res.citations_dropped,
                        "provider_ms": raw_res.elapsed_ms,
                    }
                ),
            )

            # Terminal COMPLETE event (all retrieved chunks preserved)
            yield emitter.emit(
                EventType.COMPLETE,
                "ok",
                "Run completed",
                _to_json_safe(
                    {
                        "answer": val_res.clean_text,
                        "citations": [
                            _to_json_safe(c.model_dump())
                            for c in val_res.valid_citations
                        ],
                        "chunks": chunks_json,
                        "elapsed_ms": emitter.t_ms,
                        "abstained": False,
                        "degraded": None,
                        "timings": {
                            "search_ms": search_ms,
                            "context_ms": context_ms,
                            "generate_ms": raw_res.elapsed_ms,
                        },
                    }
                ),
            )

        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            err_type_name = type(exc).__name__
            err_detail = map_exception_to_error_detail(exc, stage=current_stage)
            yield emitter.emit(
                EventType.ERROR,
                "error",
                f"Pipeline error ({err_type_name})",
                _to_json_safe(err_detail),
            )
