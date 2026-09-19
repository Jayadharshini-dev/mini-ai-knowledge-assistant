import type { DocumentRecord, HealthResponse, KnowledgeBaseStatus } from "../types/api";
import type { TraceEvent } from "../types/events";

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
