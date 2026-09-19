import type { Citation, RetrievedChunk } from "./api";

export type EventType =
  // Query stream
  | "QUERY_RECEIVED"
  | "RETRIEVAL_STARTED"
  | "RETRIEVAL_COMPLETED"
  | "EVIDENCE_SELECTED"
  | "ABSTAINED"
  | "CONTEXT_BUILT"
  | "GENERATION_STARTED"
  | "GENERATION_COMPLETED"
  | "GENERATION_SKIPPED"
  | "COMPLETE"
  | "ERROR"
  // Ingestion stream
  | "DOCUMENT_RECEIVED"
  | "DOCUMENT_DUPLICATE"
  | "TEXT_EXTRACTED"
  | "TEXT_CLEANED"
  | "CHUNKED"
  | "EMBEDDED"
  | "INDEXED";

export type EventStatus = "active" | "ok" | "warn" | "error";

export interface QueryReceivedDetail {
  question: string;
  index_size: number;
}

export interface RetrievalCompletedDetail {
  searched_vectors: number;
  index_type: string;
  returned: number;
  chunks: RetrievedChunk[];
}

export interface EvidenceSelectedDetail {
  threshold: number;
  metric: string;
  top_score: number;
  selected: number;
  rejected: number;
  decision: string;
}

export interface AbstainedDetail {
  reason: string;
  threshold: number;
  top_score: number;
  message: string;
  chunks: RetrievedChunk[];
}

export interface ContextBuiltDetail {
  passages: number;
  context_chars: number;
  token_estimate: number;
  truncated: boolean;
}

export interface GenerationStartedDetail {
  provider: string;
  model: string;
}

export interface GenerationCompletedDetail {
  citations_emitted: number;
  citations_dropped: number;
  provider_ms: number;
}

export interface GenerationSkippedDetail {
  reason: string;
  status?: string;
  mode: string;
}

export interface CompleteDetail {
  answer: string | null;
  citations: Citation[];
  chunks: RetrievedChunk[];
  elapsed_ms: number;
  abstained: boolean;
  degraded: string | null;
  timings?: {
    search_ms?: number;
    context_ms?: number;
    generate_ms?: number;
  };
}

export interface ErrorDetail {
  code: string;
  message: string;
  retryable?: boolean;
}

export interface EventBase<T extends EventType, D> {
  seq: number;
  type: T;
  status: EventStatus;
  t_ms: number;
  label: string;
  detail: D;
}

export type TraceEvent =
  | EventBase<"QUERY_RECEIVED", QueryReceivedDetail>
  | EventBase<"RETRIEVAL_STARTED", Record<string, never>>
  | EventBase<"RETRIEVAL_COMPLETED", RetrievalCompletedDetail>
  | EventBase<"EVIDENCE_SELECTED", EvidenceSelectedDetail>
  | EventBase<"ABSTAINED", AbstainedDetail>
  | EventBase<"CONTEXT_BUILT", ContextBuiltDetail>
  | EventBase<"GENERATION_STARTED", GenerationStartedDetail>
  | EventBase<"GENERATION_COMPLETED", GenerationCompletedDetail>
  | EventBase<"GENERATION_SKIPPED", GenerationSkippedDetail>
  | EventBase<"COMPLETE", CompleteDetail>
  | EventBase<"ERROR", ErrorDetail>
  | EventBase<"DOCUMENT_RECEIVED", Record<string, unknown>>
  | EventBase<"DOCUMENT_DUPLICATE", Record<string, unknown>>
  | EventBase<"TEXT_EXTRACTED", Record<string, unknown>>
  | EventBase<"TEXT_CLEANED", Record<string, unknown>>
  | EventBase<"CHUNKED", Record<string, unknown>>
  | EventBase<"EMBEDDED", Record<string, unknown>>
  | EventBase<"INDEXED", Record<string, unknown>>;


