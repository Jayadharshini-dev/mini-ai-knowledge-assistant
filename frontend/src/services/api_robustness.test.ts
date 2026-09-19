import { describe, expect, it, vi } from "vitest";
import { getHealth, sanitizeErrorMessage, streamChat, uploadDocument } from "./api";

function createMockResponse(bodyText: string, status = 200, contentType = "application/json") {
  return new Response(bodyText, {
    status,
    headers: { "Content-Type": contentType },
  });
}

describe("API Service Robustness & Error Sanitization", () => {
  it("1. sanitizeErrorMessage replaces Python tracebacks and internal paths", () => {
    const rawPythonErr = 'Traceback (most recent call last):\n  File "backend/app/routes.py", line 42, in upload\nfitz.fitz.FileNotFoundError';
    const sanitized = sanitizeErrorMessage(rawPythonErr);
    expect(sanitized).toBe("An internal server error occurred while processing the request.");
    expect(sanitized).not.toContain("Traceback");
    expect(sanitized).not.toContain(".py");
  });

  it("2. sanitizeErrorMessage preserves clean user-facing messages", () => {
    const cleanMsg = "PDF document is encrypted or password-protected";
    expect(sanitizeErrorMessage(cleanMsg)).toBe(cleanMsg);
  });

  it("3. streamChat rejects empty or whitespace question without fetch call", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    const onError = vi.fn();
    await streamChat("   ", vi.fn(), vi.fn(), vi.fn(), onError);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].code).toBe("EMPTY_QUESTION");
    vi.unstubAllGlobals();
  });

  it("4. streamChat handles fetch network rejection cleanly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Failed to fetch")));

    const onError = vi.fn();
    await streamChat("valid question", vi.fn(), vi.fn(), vi.fn(), onError);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].code).toBe("NETWORK_ERROR");
    expect(onError.mock.calls[0][0].retryable).toBe(true);
    vi.unstubAllGlobals();
  });

  it("5. uploadDocument handles non-OK HTTP error response with sanitized message", async () => {
    const errorJson = JSON.stringify({
      code: "FILE_TOO_LARGE",
      message: "File 'test.pdf' exceeds maximum allowed upload size (15 MB).",
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockResponse(errorJson, 413)));

    const onError = vi.fn();
    const testFile = new File(["dummy content"], "test.pdf", { type: "application/pdf" });
    await uploadDocument(testFile, vi.fn(), onError);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].message).toContain("test.pdf");
    vi.unstubAllGlobals();
  });

  it("6. getHealth raises formatted Error when HTTP status is not ok", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(createMockResponse("Internal Server Error", 500, "text/plain")));

    await expect(getHealth()).rejects.toThrow("Health check failed");
    vi.unstubAllGlobals();
  });
});
