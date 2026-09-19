# CONTRACTS.md
The single source of truth for the boundary between the Python backend and the TypeScript frontend of the Mini AI Knowledge Assistant. Nothing crosses the HTTP boundary that is not described here.

STATUS: frozen — Phase 05 completed.

## Rules
1. Nothing crosses the HTTP boundary that is not described here.
2. After Phase 05 this file changes only by explicit decision, and any change requires updating `backend/rag/events.py`, `frontend/src/types/events.ts` and the drift test in the same commit.
3. Event shapes and error taxonomy are frozen as of Phase 05.

---

## 1. Event envelope
Every server-sent event on every stream uses the same envelope. Only `detail` varies by `type`.

```json
{
  "seq": 4,                       // int, monotonic per request, starts at 1
  "type": "RETRIEVAL_COMPLETED",
  "status": "ok",                // "active" | "ok" | "warn" | "error"
  "t_ms": 218,                   // int, measured from request start via perf_counter. NEVER estimated.
  "label": "Searched 1,284 vectors", // short human string for the trace console
  "detail": { }                  // typed per event type, see §3
}
```

```python
# backend/rag/events.py
class TraceEvent(BaseModel):
    seq: int
    type: EventType
    status: Literal["active", "ok", "warn", "error"]
    t_ms: int
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)
```

```typescript
// frontend/src/types/events.ts
export type EventStatus = "active" | "ok" | "warn" | "error";

export interface EventBase<T extends EventType, D> {
  seq: number;
  type: T;
  status: EventStatus;
  t_ms: number;
  label: string;
  detail: D;
}
```

---

## 2. Event types
The enum is closed. Adding a member is a contract change.

### Query stream — `POST /api/chat/stream`

| TYPE | EMITTED WHEN | TERMINAL? |
| --- | --- | --- |
| `QUERY_RECEIVED` | request accepted, before any work | no |
| `RETRIEVAL_STARTED` | immediately before dense retrieval (embedding + search) | no |
| `RETRIEVAL_COMPLETED` | after FAISS search returns retrieved candidates | no |
| `EVIDENCE_SELECTED` | after the relevance gate decides | no |
| `ABSTAINED` | gate decided no passage clears the threshold | no |
| `CONTEXT_BUILT` | numbered context assembled | no |
| `GENERATION_STARTED` | immediately before the provider call | no |
| `GENERATION_COMPLETED` | provider returned and citations validated | no |
| `GENERATION_SKIPPED` | no provider configured / provider degraded | no |
| `COMPLETE` | final event, carries the answer/outcome payload | **yes** |
| `ERROR` | any typed failure | **yes** |

Terminal Event Invariant. Every query stream run ends with exactly one terminal event: `COMPLETE` for successful, abstained, or degraded outcomes, or `ERROR` for unhandled failures. `ABSTAINED` and `GENERATION_SKIPPED` are non-terminal informational events followed by `COMPLETE`.

Pipeline Execution Rules:
1. Per-Run EventEmitter: `EventEmitter` is instantiated locally inside `run()` to guarantee independent `seq` and `t_ms` streams for concurrent runs.
2. Non-Blocking Execution: All blocking operations (`retriever.retrieve`, `provider.generate`) use `await asyncio.to_thread(...)`. Query embedding is encapsulated inside `DenseRetriever.retrieve()`.
3. Threshold Injection: Calibrated threshold is bound at instantiation via `functools.partial(evaluate_relevance, threshold=settings.RELEVANCE_THRESHOLD)`. `RagPipeline` remains setting-independent.
4. Pure JSON Serializability: Every detail dictionary contains pure Python primitives (float, int, str, bool, list, dict) and passes `json.dumps(event.to_dict())` without custom encoders.

### Ingestion stream — `POST /api/documents`

| TYPE | EMITTED WHEN | TERMINAL? |
| --- | --- | --- |
| `DOCUMENT_RECEIVED` | upload accepted, hash computed | no |
| `DOCUMENT_DUPLICATE` | hash already in the registry | **yes** |
| `TEXT_EXTRACTED` | PyMuPDF finished | no |
| `TEXT_CLEANED` | cleaner finished | no |
| `CHUNKED` | chunker finished | no |
| `EMBEDDED` | all chunk vectors computed | no |
| `INDEXED` | FAISS index and manifest written to disk | no |
| `COMPLETE` | terminal, carries final document record | **yes** |
| `ERROR` | typed failure | **yes** |

Ordering guarantee. Events arrive in the order listed. `tests/test_pipeline.py` asserts the exact sequence for: the happy path, the abstention path, the no-provider path, and the error path.

---

## 3. Event detail payloads
Only non-empty payloads are listed. Every other type carries `detail: {}`.

### QUERY_RECEIVED
```json
{ "question": "string", "index_size": 1284 }
```

### RETRIEVAL_COMPLETED
```json
{
  "searched_vectors": 1284,
  "index_type": "IndexFlatIP",
  "returned": 4,
  "chunks": [ /* RetrievedChunk[] */ ]
}
```

### EVIDENCE_SELECTED
```json
{
  "threshold": 0.67,
  "metric": "cosine",
  "top_score": 0.81,
  "selected": 3,
  "rejected": 1,
  "decision": "proceed"
}
```

### ABSTAINED (terminal)
```json
{
  "reason": "no_passage_above_threshold",
  "threshold": 0.67,
  "top_score": 0.11,
  "message": "The knowledge base does not contain information relevant to this question.",
  "chunks": [ /* sub-threshold RetrievedChunk[] */ ]
}
```

### CONTEXT_BUILT
```json
{ "passages": 3, "context_chars": 2140, "token_estimate": 512, "truncated": false }
```

### GENERATION_STARTED
```json
{ "provider": "gemini", "model": "gemini-2.5-flash" }
```

### GENERATION_COMPLETED
```json
{ "citations_emitted": 3, "citations_dropped": 0, "provider_ms": 1847 }
```

### GENERATION_SKIPPED (terminal)
```json
{ "reason": "no_provider_configured", "mode": "evidence_only" }
```

### COMPLETE (terminal)
```json
{
  "answer": "string | null",
  "citations": [ /* Citation[] */ ],
  "chunks": [ /* RetrievedChunk[] */ ],
  "elapsed_ms": 2810,
  "timings": { "embed_ms": 41, "search_ms": 3, "context_ms": 8, "generate_ms": 1847 }
}
```

### ERROR (terminal)
```json
{ "code": "PROVIDER_UNAVAILABLE", "message": "human sentence", "retryable": true }
```

Citation invariant. Every entry in `citations` references a `chunk_id` present in `chunks`. A marker the model emits with no matching chunk is dropped and counted in `citations_dropped`. The frontend may assume this invariant holds.

---

## 4. Shared object shapes

### Chunk
```json
{
  "chunk_id": "ml_notes__p042__c0118",
  "doc_id": "a3f91c8e2b04",
  "document": "Machine_Learning.pdf",
  "page": 42,
  "chunk_index": 118,
  "text": "...",
  "char_count": 874,
  "token_estimate": 196
}
```

### RetrievedChunk
```json
{
  "chunk": { /* Chunk */ },
  "score": 0.8213,
  "rank": 1,
  "above_threshold": true
}
```

### Citation
```json
{
  "n": 1,
  "chunk_id": "ml_notes__p042__c0118",
  "document": "Machine_Learning.pdf",
  "page": 42
}
```

Score is labelled **"similarity (cosine)"** wherever it appears in the UI. It is never called confidence, probability, accuracy or truth.

---

## 5. HTTP endpoints

| METHOD | PATH | REQUEST | RESPONSE |
| --- | --- | --- | --- |
| GET | `/api/health` | — | 200 resolved config (no secrets) |
| GET | `/api/knowledge-base/status` | — | 200 §6 |
| GET | `/api/documents` | — | 200 `{ "documents": DocumentRecord[] }` |
| POST | `/api/documents` | `multipart/form-data`, field `file` | 200 `text/event-stream` — ingestion events |
| POST | `/api/chat/stream` | §7 | 200 `text/event-stream` — query events |
| POST | `/api/chat` | §7 | 200 `{ ...COMPLETE.detail, "trace": TraceEvent[] }` |

No job ids, no polling. `POST /api/documents` streams its own progress on the upload response.

SSE framing. One event per frame, `event:` line always present:
```
event: trace
data: {"seq":1,"type":"QUERY_RECEIVED",...}

event: done
data: {"seq":11,"type":"COMPLETE",...}
```

Required response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`. No gzip middleware on streaming routes.

---

## 6. Knowledge-base status
```json
{
  "state": "ready",
  "documents": 3,
  "chunks": 1284,
  "vector_dim": 384,
  "index_type": "IndexFlatIP",
  "index_size_bytes": 1971456,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash",
  "llm_available": true,
  "top_k": 4,
  "relevance_threshold": 0.67,
  "retriever": "dense"
}
```

---

## 7. Chat request
```json
{
  "question": "string",
  "history": [
    { "role": "user", "content": "string" }
  ]
}
```

---

## 8. Error codes

| CODE | MEANING | HTTP / EVENT |
| --- | --- | --- |
| `EMPTY_QUESTION` | question blank after trim | 400 |
| `NO_DOCUMENTS` | nothing indexed yet | 409 |
| `INVALID_PDF` | not a PDF or unparseable | 400 |
| `ENCRYPTED_PDF` | password protected | 400 |
| `NO_EXTRACTABLE_TEXT` | likely a scan; OCR out of scope | 422 |
| `DUPLICATE_DOCUMENT` | content hash already indexed | 409 |
| `FILE_TOO_LARGE` | above MAX_UPLOAD_MB | 413 |
| `INDEX_MODEL_MISMATCH` | manifest model ≠ configured model | 500 |
| `INDEX_DIMENSION_MISMATCH` | manifest dimension ≠ configured dimension | 500 |
| `INDEX_TYPE_MISMATCH` | manifest index type ≠ configured type | 500 |
| `INDEX_METADATA_MISMATCH` | vector count ≠ metadata/manifest count | 500 |
| `EMBEDDING_FAILED` | embedder raised | ERROR event |
| `RETRIEVAL_FAILED` | FAISS raised | ERROR event |
| `PROVIDER_UNAVAILABLE` | network / 5xx / timeout | ERROR event |
| `PROVIDER_RATE_LIMITED` | 429 — degrade to evidence-only | ERROR event, retryable: true |
| `PROVIDER_AUTH` | key missing or rejected | ERROR event |
| `INTERNAL_ERROR` | unhandled server exception | ERROR event |

---

## 9. Drift test
`tests/test_contract.py` must assert that the Python enum and the TypeScript union list identical members.

---

## 10. Change log
- **2026-09-17**: Skeleton authored pre-Phase 00.
