import { describe, expect, it, vi } from "vitest";
import { streamChat } from "./api";
import type { TraceEvent } from "../types/events";

/**
 * Helper to create a mocked Fetch response with a custom ReadableStream body
 */
function createMockStreamResponse(chunks: string[], status = 200) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });

  return new Response(stream, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("streamChat Production SSE Stream Integration", () => {
  it("1. handles one complete SSE frame", async () => {
    const frame = `event: trace\ndata: {"seq":1,"type":"QUERY_RECEIVED","status":"ok","t_ms":5,"label":"Query received","detail":{"question":"test","index_size":10}}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame])));

    const onTrace = vi.fn();
    const onComplete = vi.fn();
    const onAbstained = vi.fn();
    const onError = vi.fn();

    await streamChat("test question", onTrace, onComplete, onAbstained, onError);

    expect(onTrace).toHaveBeenCalledTimes(1);
    const traceArg = onTrace.mock.calls[0][0] as TraceEvent;
    expect(traceArg.type).toBe("QUERY_RECEIVED");
    expect(traceArg.seq).toBe(1);
    vi.unstubAllGlobals();
  });

  it("2. handles JSON/frame split across network chunks (buffer carry-over)", async () => {
    const chunk1 = `event: trace\ndata: {"seq":1,"type":"QUERY_RECEIVED","status":"ok","t_ms":2,"label":"Query received","detail":{}}\n\nevent: trace\ndata: {"seq":2,"type":"RET`;
    const chunk2 = `RIEVAL_STARTED","status":"ok","t_ms":10,"label":"Searching index","detail":{}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([chunk1, chunk2])));

    const onTrace = vi.fn();
    await streamChat("test question", onTrace, vi.fn(), vi.fn(), vi.fn());

    expect(onTrace).toHaveBeenCalledTimes(2);
    expect((onTrace.mock.calls[0][0] as TraceEvent).type).toBe("QUERY_RECEIVED");
    expect((onTrace.mock.calls[1][0] as TraceEvent).type).toBe("RETRIEVAL_STARTED");
    vi.unstubAllGlobals();
  });

  it("3. handles multiple frames in one network chunk", async () => {
    const chunk =
      `event: trace\ndata: {"seq":1,"type":"QUERY_RECEIVED","status":"ok","t_ms":1,"label":"Received","detail":{}}\n\n` +
      `event: trace\ndata: {"seq":2,"type":"RETRIEVAL_STARTED","status":"ok","t_ms":5,"label":"Searching","detail":{}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([chunk])));

    const onTrace = vi.fn();
    await streamChat("test question", onTrace, vi.fn(), vi.fn(), vi.fn());

    expect(onTrace).toHaveBeenCalledTimes(2);
    vi.unstubAllGlobals();
  });

  it("4. handles terminal COMPLETE event", async () => {
    const frame = `event: done\ndata: {"seq":8,"type":"COMPLETE","status":"ok","t_ms":120,"label":"Complete","detail":{"answer":"Grounded answer","citations":[],"chunks":[],"elapsed_ms":120,"abstained":false,"degraded":null}}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame])));

    const onComplete = vi.fn();
    const onError = vi.fn();

    await streamChat("test question", vi.fn(), onComplete, vi.fn(), onError);

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0].answer).toBe("Grounded answer");
    expect(onError).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("5. handles terminal ERROR event", async () => {
    const frame = `event: done\ndata: {"seq":5,"type":"ERROR","status":"error","t_ms":45,"label":"Error","detail":{"code":"PROVIDER_UNAVAILABLE","message":"Gemini offline","retryable":true}}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame])));

    const onError = vi.fn();
    await streamChat("test question", vi.fn(), vi.fn(), vi.fn(), onError);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].code).toBe("PROVIDER_UNAVAILABLE");
    expect(onError.mock.calls[0][0].retryable).toBe(true);
    vi.unstubAllGlobals();
  });

  it("6. handles ABSTAINED followed by COMPLETE", async () => {
    const frame1 = `event: trace\ndata: {"seq":4,"type":"ABSTAINED","status":"warn","t_ms":30,"label":"Abstained","detail":{"reason":"no_passage_above_threshold","threshold":0.67,"top_score":0.1,"message":"Knowledge base does not contain info","chunks":[]}}\n\n`;
    const frame2 = `event: done\ndata: {"seq":5,"type":"COMPLETE","status":"ok","t_ms":35,"label":"Complete","detail":{"answer":null,"citations":[],"chunks":[],"elapsed_ms":35,"abstained":true,"degraded":null}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame1, frame2])));

    const onAbstained = vi.fn();
    const onComplete = vi.fn();

    await streamChat("test question", vi.fn(), onComplete, onAbstained, vi.fn());

    expect(onAbstained).toHaveBeenCalledTimes(1);
    expect(onAbstained.mock.calls[0][0].threshold).toBe(0.67);
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0].abstained).toBe(true);
    vi.unstubAllGlobals();
  });

  it("7. handles GENERATION_SKIPPED followed by COMPLETE", async () => {
    const frame1 = `event: trace\ndata: {"seq":6,"type":"GENERATION_SKIPPED","status":"warn","t_ms":40,"label":"Skipped","detail":{"reason":"no_provider_configured","mode":"evidence_only"}}\n\n`;
    const frame2 = `event: done\ndata: {"seq":7,"type":"COMPLETE","status":"ok","t_ms":45,"label":"Complete","detail":{"answer":null,"citations":[],"chunks":[],"elapsed_ms":45,"abstained":false,"degraded":"no_provider"}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame1, frame2])));

    const onTrace = vi.fn();
    const onComplete = vi.fn();

    await streamChat("test question", onTrace, onComplete, vi.fn(), vi.fn());

    expect(onTrace).toHaveBeenCalledTimes(2);
    expect((onTrace.mock.calls[0][0] as TraceEvent).type).toBe("GENERATION_SKIPPED");
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0].degraded).toBe("no_provider");
    vi.unstubAllGlobals();
  });

  it("8. handles malformed JSON without crashing stream", async () => {
    const frame1 = `event: trace\ndata: {invalid_json_payload\n\n`;
    const frame2 = `event: done\ndata: {"seq":2,"type":"COMPLETE","status":"ok","t_ms":10,"label":"Complete","detail":{"answer":"ok","citations":[],"chunks":[],"elapsed_ms":10,"abstained":false,"degraded":null}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame1, frame2])));

    const onComplete = vi.fn();
    await streamChat("test question", vi.fn(), onComplete, vi.fn(), vi.fn());

    expect(onComplete).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });

  it("9. handles block with no data field gracefully", async () => {
    const frame1 = `event: trace\ncomment: ping without data\n\n`;
    const frame2 = `event: done\ndata: {"seq":2,"type":"COMPLETE","status":"ok","t_ms":10,"label":"Complete","detail":{"answer":"ok","citations":[],"chunks":[],"elapsed_ms":10,"abstained":false,"degraded":null}}\n\n`;

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame1, frame2])));

    const onComplete = vi.fn();
    await streamChat("test question", vi.fn(), onComplete, vi.fn(), vi.fn());

    expect(onComplete).toHaveBeenCalledTimes(1);
    vi.unstubAllGlobals();
  });

  it("10. handles backend event: error framing", async () => {
    const frame = `event: error\ndata: {"code":"INTERNAL_ERROR","message":"Server crashed","retryable":false}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame])));

    const onError = vi.fn();
    await streamChat("test question", vi.fn(), vi.fn(), vi.fn(), onError);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].code).toBe("INTERNAL_ERROR");
    expect(onError.mock.calls[0][0].message).toBe("Server crashed");
    vi.unstubAllGlobals();
  });

  it("11. handles stream ending without terminal event", async () => {
    const frame = `event: trace\ndata: {"seq":1,"type":"QUERY_RECEIVED","status":"ok","t_ms":1,"label":"Received","detail":{}}\n\n`;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockStreamResponse([frame])));

    const onError = vi.fn();
    await streamChat("test question", vi.fn(), vi.fn(), vi.fn(), onError);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].code).toBe("STREAM_INTERRUPTED");
    vi.unstubAllGlobals();
  });
});
