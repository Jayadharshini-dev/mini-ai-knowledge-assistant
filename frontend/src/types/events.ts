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

export interface TraceEvent {
  seq: number;
  type: EventType;
  status: EventStatus;
  t_ms: number;
  label: string;
  detail: Record<string, unknown>;
}
