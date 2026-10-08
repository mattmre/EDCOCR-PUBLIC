import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearApiKey, setApiKey } from "@/lib/auth";
import { __testing, buildJobWebSocketUrl, useJobWebSocket } from "@/lib/ws-client";

interface MockSocket extends WebSocket {
  readyState: number;
  __mockSend: ReturnType<typeof vi.fn>;
  __sentMessages: string[];
  __protocols: string | string[] | undefined;
  __mockOpen: (echoedProtocol?: string) => void;
  __mockMessage: (data: unknown) => void;
  __mockClose: (code?: number, reason?: string) => void;
  __mockError: () => void;
}

function makeMockWebSocketCtor(instances: MockSocket[]) {
  const Mock = vi.fn(function (
    this: MockSocket,
    _url: string | URL,
    protocols?: string | string[]
  ) {
    const inst: MockSocket = this;
    inst.readyState = 0; // CONNECTING
    inst.__sentMessages = [];
    inst.__protocols = protocols;
    const offeredProtocol =
      typeof protocols === "string" ? protocols : (protocols?.[0] ?? "");
    Object.defineProperty(inst, "protocol", {
      configurable: true,
      value: offeredProtocol,
    });
    inst.__mockSend = vi.fn((payload: string) => {
      inst.__sentMessages.push(payload);
    });
    inst.send = inst.__mockSend as unknown as typeof inst.send;
    inst.close = vi.fn((code?: number, reason?: string) => {
      inst.readyState = 3;
      inst.onclose?.(new CloseEvent("close", { code: code ?? 1000, reason: reason ?? "" }));
    }) as unknown as typeof inst.close;
    inst.__mockOpen = (echoedProtocol = offeredProtocol) => {
      Object.defineProperty(inst, "protocol", {
        configurable: true,
        value: echoedProtocol,
      });
      inst.readyState = 1;
      inst.onopen?.(new Event("open"));
    };
    inst.__mockMessage = (data: unknown) => {
      const text = typeof data === "string" ? data : JSON.stringify(data);
      inst.onmessage?.(new MessageEvent("message", { data: text }));
    };
    inst.__mockClose = (code = 1000, reason = "") => {
      inst.readyState = 3;
      inst.onclose?.(new CloseEvent("close", { code, reason }));
    };
    inst.__mockError = () => {
      inst.onerror?.(new Event("error"));
    };
    instances.push(inst);
  }) as unknown as typeof WebSocket;
  return Mock;
}

describe("buildJobWebSocketUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("resolves the Helm root-relative default against the page origin", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "/");
    const expectedProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    expect(buildJobWebSocketUrl("job_abc")).toBe(
      `${expectedProtocol}//${window.location.host}/ws/jobs/job_abc`
    );
  });

  it("preserves an ingress path prefix for a root-relative base", () => {
    const expectedProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    expect(buildJobWebSocketUrl("job_abc", "/ocr/")).toBe(
      `${expectedProtocol}//${window.location.host}/ocr/ws/jobs/job_abc`
    );
  });

  it("converts http -> ws", () => {
    expect(buildJobWebSocketUrl("job_abc", "http://localhost:8000")).toBe(
      "ws://localhost:8000/ws/jobs/job_abc"
    );
  });

  it("converts https -> wss and trims trailing slash", () => {
    expect(buildJobWebSocketUrl("job_abc", "https://api.example.com/")).toBe(
      "wss://api.example.com/ws/jobs/job_abc"
    );
  });

  it("preserves an explicit absolute WebSocket override", () => {
    expect(buildJobWebSocketUrl("job_abc", "wss://socket.example.com/ocr/")).toBe(
      "wss://socket.example.com/ocr/ws/jobs/job_abc"
    );
  });

  it("encodes the job id", () => {
    expect(buildJobWebSocketUrl("job/with space", "http://x")).toContain(
      "/ws/jobs/job%2Fwith%20space"
    );
  });

  it("rejects malformed or unsupported bases", () => {
    expect(() => buildJobWebSocketUrl("job_abc", "not-a-url")).toThrow(
      "Invalid WebSocket base URL"
    );
    expect(() => buildJobWebSocketUrl("job_abc", "ftp://api.example.com")).toThrow(
      "Invalid WebSocket base URL"
    );
  });

  it("rejects a relative base when no page origin exists", () => {
    vi.stubGlobal("window", undefined);
    expect(() => buildJobWebSocketUrl("job_abc", "/")).toThrow("Invalid WebSocket base URL");
  });
});

describe("nextBackoffDelay", () => {
  it("starts at the base interval and caps at the max", () => {
    const d0 = __testing.nextBackoffDelay(0);
    expect(d0).toBeGreaterThanOrEqual(__testing.RECONNECT_BASE_MS * 0.8);
    expect(d0).toBeLessThanOrEqual(__testing.RECONNECT_BASE_MS * 1.2);

    const d10 = __testing.nextBackoffDelay(10);
    expect(d10).toBeLessThanOrEqual(__testing.RECONNECT_MAX_MS * 1.2);
    expect(d10).toBeGreaterThanOrEqual(__testing.RECONNECT_MAX_MS * 0.8);
  });
});

describe("useJobWebSocket", () => {
  let instances: MockSocket[] = [];
  let MockWS: typeof WebSocket;

  beforeEach(() => {
    instances = [];
    MockWS = makeMockWebSocketCtor(instances);
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it("opens a socket and sends the auth frame on open", async () => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    expect(result.current.status).toBe("connecting");
    expect(instances[0].__protocols).toBe("ocr-local.auth.frame.v1");

    act(() => instances[0].__mockOpen());
    expect(instances[0].__sentMessages).toHaveLength(1);
    const frame = JSON.parse(instances[0].__sentMessages[0]);
    // Mock-socket fixture value only; never a real credential.
    expect(frame).toEqual({ type: "auth", api_key: "test-key" }); // pragma: allowlist secret

    unmount();
  });

  it("uses the HttpOnly cookie mode without sending an auth frame", async () => {
    vi.stubEnv("NEXT_PUBLIC_OCR_HTTPONLY_SESSION", "true");
    clearApiKey();
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    expect(instances[0].__protocols).toBe("ocr-local.auth.cookie.v1");
    act(() => instances[0].__mockOpen());

    expect(result.current.status).toBe("authenticating");
    expect(instances[0].__sentMessages).toHaveLength(0);
    unmount();
  });

  it("keeps an explicit key in frame mode when HttpOnly cookies are enabled", async () => {
    vi.stubEnv("NEXT_PUBLIC_OCR_HTTPONLY_SESSION", "true");
    const { unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", {
        webSocketImpl: MockWS,
        baseUrl: "http://x",
        apiKey: "explicit-test-key",
      })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    expect(instances[0].__protocols).toBe("ocr-local.auth.frame.v1");
    act(() => instances[0].__mockOpen());
    expect(JSON.parse(instances[0].__sentMessages[0])).toEqual({
      type: "auth",
      api_key: "explicit-test-key",
    });
    unmount();
  });

  it("does not fall back to a cookie for an explicitly empty frame credential", async () => {
    vi.stubEnv("NEXT_PUBLIC_OCR_HTTPONLY_SESSION", "true");
    const { unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", {
        webSocketImpl: MockWS,
        baseUrl: "http://x",
        apiKey: "",
      })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    expect(instances[0].__protocols).toBe("ocr-local.auth.frame.v1");
    act(() => instances[0].__mockOpen());
    expect(JSON.parse(instances[0].__sentMessages[0])).toEqual({
      type: "auth",
      api_key: "",
    });
    unmount();
  });

  it("closes without sending credentials when the server does not echo auth mode", async () => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen(""));

    expect(result.current.status).toBe("error");
    expect(instances[0].__sentMessages).toHaveLength(0);
    expect(instances[0].close).toHaveBeenCalledWith(
      4002,
      "WebSocket authentication protocol mismatch"
    );
    unmount();
  });

  it("transitions to open after the connected message", async () => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    act(() =>
      instances[0].__mockMessage({
        type: "connected",
        job_id: "job_aaaaaaaaaaaa",
        status: "queued",
      })
    );
    await waitFor(() => expect(result.current.status).toBe("open"));
    expect(result.current.lastMessage).toMatchObject({ type: "connected" });
    unmount();
  });

  it("buffers progress messages and exposes the latest one", async () => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    act(() =>
      instances[0].__mockMessage({ type: "connected", job_id: "j", status: "queued" })
    );
    act(() =>
      instances[0].__mockMessage({
        type: "progress",
        job_id: "j",
        status: "processing",
        pages_completed: 3,
        total_pages: 10,
        percent: 30,
      })
    );
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    expect(result.current.lastMessage?.type).toBe("progress");
    unmount();
  });

  it("auto-reconnects with backoff after an unexpected close", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { result, unmount } = renderHook(() =>
        useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
      );
      await waitFor(() => expect(instances.length).toBe(1));
      act(() => instances[0].__mockOpen());
      act(() => instances[0].__mockClose(1011, "Transient server failure"));

      expect(result.current.status).toBe("reconnecting");

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2_000);
      });
      await waitFor(() => expect(instances.length).toBeGreaterThanOrEqual(2));

      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not auto-reconnect after authentication or session invalidation", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { result, unmount } = renderHook(() =>
        useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
      );
      await waitFor(() => expect(instances.length).toBe(1));
      act(() => instances[0].__mockOpen());
      act(() => instances[0].__mockClose(4001, "Session terminated"));

      expect(result.current.status).toBe("error");
      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      expect(instances).toHaveLength(1);

      // Explicit user recovery remains available after credentials change.
      act(() => result.current.reconnect());
      await waitFor(() => expect(instances).toHaveLength(2));
      expect(result.current.status).toBe("connecting");
      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("ignores malformed JSON payloads", async () => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    act(() => instances[0].__mockMessage("{not-json"));
    expect(result.current.messages).toHaveLength(0);
    unmount();
  });

  it("cleans up on unmount", async () => {
    const { unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    unmount();
    expect(instances[0].close).toHaveBeenCalled();
  });
});
