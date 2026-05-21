import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setApiKey } from "@/lib/auth";
import { __testing, buildJobWebSocketUrl, useJobWebSocket } from "@/lib/ws-client";

interface MockSocket extends WebSocket {
  __mockSend: ReturnType<typeof vi.fn>;
  __sentMessages: string[];
  __mockOpen: => void;
  __mockMessage: (data: unknown) => void;
  __mockClose: => void;
  __mockError: => void;
}

function makeMockWebSocketCtor(instances: MockSocket[]) {
  const Mock = vi.fn(function (this: MockSocket, _url: string) {
    const inst: MockSocket = this;
    inst.readyState = 0; // CONNECTING
    inst.__sentMessages = [];
    inst.__mockSend = vi.fn((payload: string) => {
      inst.__sentMessages.push(payload);
    });
    inst.send = inst.__mockSend as unknown as typeof inst.send;
    inst.close = vi.fn(() => {
      inst.readyState = 3;
      inst.onclose?.(new CloseEvent("close"));
    }) as unknown as typeof inst.close;
    inst.__mockOpen = => {
      inst.readyState = 1;
      inst.onopen?.(new Event("open"));
    };
    inst.__mockMessage = (data: unknown) => {
      const text = typeof data === "string" ? data : JSON.stringify(data);
      inst.onmessage?.(new MessageEvent("message", { data: text }));
    };
    inst.__mockClose = => {
      inst.readyState = 3;
      inst.onclose?.(new CloseEvent("close"));
    };
    inst.__mockError = => {
      inst.onerror?.(new Event("error"));
    };
    instances.push(inst);
  }) as unknown as typeof WebSocket;
  return Mock;
}

describe("buildJobWebSocketUrl", => {
  it("converts http -> ws", => {
    expect(buildJobWebSocketUrl("job_abc", "http://localhost:8000")).toBe(
      "ws://localhost:8000/ws/jobs/job_abc"
    );
  });

  it("converts https -> wss and trims trailing slash", => {
    expect(buildJobWebSocketUrl("job_abc", "https://api.example.com/")).toBe(
      "wss://api.example.com/ws/jobs/job_abc"
    );
  });

  it("encodes the job id", => {
    expect(buildJobWebSocketUrl("job/with space", "http://x")).toContain(
      "/ws/jobs/job%2Fwith%20space"
    );
  });
});

describe("nextBackoffDelay", => {
  it("starts at the base interval and caps at the max", => {
    const d0 = __testing.nextBackoffDelay(0);
    expect(d0).toBeGreaterThanOrEqual(__testing.RECONNECT_BASE_MS * 0.8);
    expect(d0).toBeLessThanOrEqual(__testing.RECONNECT_BASE_MS * 1.2);

    const d10 = __testing.nextBackoffDelay(10);
    expect(d10).toBeLessThanOrEqual(__testing.RECONNECT_MAX_MS * 1.2);
    expect(d10).toBeGreaterThanOrEqual(__testing.RECONNECT_MAX_MS * 0.8);
  });
});

describe("useJobWebSocket", => {
  let instances: MockSocket[] = [];
  let MockWS: typeof WebSocket;

  beforeEach(() => {
    instances = [];
    MockWS = makeMockWebSocketCtor(instances);
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("opens a socket and sends the auth frame on open", async => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );

    await waitFor(() => expect(instances.length).toBe(1));
    expect(result.current.status).toBe("connecting");

    act(() => instances[0].__mockOpen());
    expect(instances[0].__sentMessages).toHaveLength(1);
    const frame = JSON.parse(instances[0].__sentMessages[0]);
    expect(frame).toEqual({ type: "auth", api_key: "test-key" });

    unmount();
  });

  it("transitions to open after the connected message", async => {
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

  it("buffers progress messages and exposes the latest one", async => {
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

  it("auto-reconnects with backoff after an unexpected close", async => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const { result, unmount } = renderHook(() =>
        useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
      );
      await waitFor(() => expect(instances.length).toBe(1));
      act(() => instances[0].__mockOpen());
      act(() => instances[0].__mockClose());

      expect(result.current.status).toBe("reconnecting");

      await act(async => {
        await vi.advanceTimersByTimeAsync(2_000);
      });
      await waitFor(() => expect(instances.length).toBeGreaterThanOrEqual(2));

      unmount();
    } finally {
      vi.useRealTimers();
    }
  });

  it("ignores malformed JSON payloads", async => {
    const { result, unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    act(() => instances[0].__mockMessage("{not-json"));
    expect(result.current.messages).toHaveLength(0);
    unmount();
  });

  it("cleans up on unmount", async => {
    const { unmount } = renderHook(() =>
      useJobWebSocket("job_aaaaaaaaaaaa", { webSocketImpl: MockWS, baseUrl: "http://x" })
    );
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].__mockOpen());
    unmount();
    expect(instances[0].close).toHaveBeenCalled();
  });
});
