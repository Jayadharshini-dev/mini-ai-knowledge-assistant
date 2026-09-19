import type { ApiError, DocumentRecord, HealthResponse, KnowledgeBaseStatus } from "../types/api";
import type { AbstainedDetail, CompleteDetail, ErrorDetail, TraceEvent } from "../types/events";


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function getKnowledgeBaseStatus(): Promise<KnowledgeBaseStatus> {
  const response = await fetch(`${API_BASE_URL}/api/kb/status`);
  if (!response.ok) {
    throw new Error(`KB status check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function getDocuments(): Promise<{ documents: DocumentRecord[] }> {
  const response = await fetch(`${API_BASE_URL}/api/documents`);
  if (!response.ok) {
    throw new Error(`Failed to list documents: ${response.statusText}`);
  }
  return response.json();
}

export async function uploadDocument(
  file: File,
  onEvent: (event: TraceEvent) => void,
  onError: (error: Error) => void
): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE_URL}/api/documents`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Upload failed with status ${response.status}`);
    }

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        if (!block.trim()) continue;
        const lines = block.split("\n");
        let dataStr = "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            dataStr = line.replace("data: ", "").trim();
          }
        }

        if (dataStr) {
          try {
            const parsedEvent = JSON.parse(dataStr) as TraceEvent;
            onEvent(parsedEvent);
          } catch (e) {
            console.error("Failed to parse SSE event payload:", e);
          }
        }
      }
    }
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

export function parseSseBlock(block: string): { eventName?: string; dataStr?: string } {
  const lines = block.split("\n");
  let eventName: string | undefined;
  let dataStr: string | undefined;
  for (const line of lines) {
    if (line.startsWith("event: ")) {
      eventName = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      dataStr = line.slice(6).trim();
    }
  }
  return { eventName, dataStr };
}

export async function streamChat(
  question: string,
  onTraceEvent: (event: TraceEvent) => void,
  onComplete: (completeDetail: CompleteDetail) => void,
  onAbstained: (abstainedDetail: AbstainedDetail) => void,
  onError: (error: ApiError) => void,
  signal?: AbortSignal
): Promise<void> {
  const rawQuestion = question.trim();
  if (!rawQuestion) {
    onError({
      code: "EMPTY_QUESTION",
      message: "Question cannot be empty.",
      retryable: false,
    });
    return;
  }

  let receivedTerminalEvent = false;

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: rawQuestion }),
      signal,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      onError({
        code: errData.code || "HTTP_ERROR",
        message: errData.message || `Request failed with status ${response.status}`,
        retryable: response.status >= 500,
      });
      return;
    }

    if (!response.body) {
      onError({
        code: "STREAM_ERROR",
        message: "Response stream is empty.",
        retryable: false,
      });
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        if (!block.trim()) continue;
        const { eventName, dataStr } = parseSseBlock(block);

        if (!dataStr) continue;

        if (eventName === "error") {
          try {
            const errPayload = JSON.parse(dataStr) as ErrorDetail;
            receivedTerminalEvent = true;
            onError({
              code: errPayload.code || "INTERNAL_ERROR",
              message: errPayload.message || "Streaming server error occurred.",
              retryable: errPayload.retryable ?? false,
            });
          } catch {
            receivedTerminalEvent = true;
            onError({
              code: "INTERNAL_ERROR",
              message: "Malformed server error event payload.",
              retryable: false,
            });
          }
          continue;
        }

        try {
          const parsedData = JSON.parse(dataStr);
          if (
            parsedData &&
            typeof parsedData === "object" &&
            typeof parsedData.type === "string" &&
            typeof parsedData.seq === "number" &&
            typeof parsedData.t_ms === "number"
          ) {
            const traceEvent = parsedData as TraceEvent;
            onTraceEvent(traceEvent);

            if (traceEvent.type === "ABSTAINED") {
              onAbstained(traceEvent.detail as AbstainedDetail);
            } else if (traceEvent.type === "COMPLETE") {
              receivedTerminalEvent = true;
              onComplete(traceEvent.detail as CompleteDetail);
            } else if (traceEvent.type === "ERROR") {
              receivedTerminalEvent = true;
              const errDetail = traceEvent.detail as ErrorDetail;
              onError({
                code: errDetail.code || "PIPELINE_ERROR",
                message: errDetail.message || "An error occurred during query execution.",
                retryable: errDetail.retryable,
              });
            }
          }
        } catch (e) {
          console.error("Failed to parse SSE trace event payload:", e);
        }
      }
    }

    if (!receivedTerminalEvent && !signal?.aborted) {
      onError({
        code: "STREAM_INTERRUPTED",
        message: "The event stream ended unexpectedly before completion.",
        retryable: true,
      });
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    onError({
      code: "NETWORK_ERROR",
      message: err instanceof Error ? err.message : String(err),
      retryable: true,
    });
  }
}


