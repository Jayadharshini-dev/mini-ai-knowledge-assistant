import type { TraceEvent } from "./events";

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  document: string;
  page: number;
  chunk_index: number;
  text: string;
  char_count: number;
  token_estimate: number;
}

export interface RetrievedChunk {
  chunk: Chunk;
  score: number;
  rank: number;
  above_threshold: boolean;
}

export interface Citation {
  n: number;
  chunk_id: string;
  document: string;
  page: number;
}

export interface DocumentRecord {
  doc_id: string;
  filename: string;
  pages: number;
  chunks: number;
  indexed_at: string;
  status: "indexed" | "failed";
}

export interface KnowledgeBaseStatus {
  state: "empty" | "indexing" | "ready" | "error";
  documents: number;
  chunks: number;
  vector_dim: number;
  index_type: string;
  index_size_bytes: number;
  embedding_model: string;
  llm_provider: string;
  llm_model: string;
  llm_available: boolean;
  top_k: number;
  relevance_threshold: number;
  retriever: "dense" | "hybrid";
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  history?: ChatMessage[];
}

export interface ChatResponse {
  answer: string | null;
  citations: Citation[];
  chunks: RetrievedChunk[];
  elapsed_ms: number;
  timings?: {
    search_ms: number;
    context_ms: number;
    generate_ms: number;
  };
  trace: TraceEvent[];
}

export interface HealthResponse {
  status: string;
  config: {
    CHUNK_SIZE: number;
    CHUNK_OVERLAP: number;
    MAX_UPLOAD_MB: number;
    EMBEDDING_MODEL: string;
    VECTOR_DIM: number;
    INDEX_TYPE: string;
    TOP_K: number;
    RELEVANCE_THRESHOLD: number;
    LLM_PROVIDER: string;
    LLM_MODEL: string;
    GEMINI_API_KEY_SET: boolean;
  };
}

export interface ApiError {
  code: string;
  message: string;
  retryable?: boolean;
}
